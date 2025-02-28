import torch
import torch.nn.functional as F
import json
import argparse
import random
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings
import math
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from torch.utils.data import Dataset
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from utils.qa_dataset import QADataset

def set_seed(seed: int = 42):
    """
    Set all relevant random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(args):
    # ======================================== #
    # ===Feature-based Similarity Functions=== #
    # ======================================== #
    def compute_ngram_similarity(questions, n=2):
        def get_ngrams(text, n):
            tokens = text.lower().split()
            return set(zip(*[tokens[i:] for i in range(n)])) if len(tokens) >= n else set()

        N = len(questions)
        similarity_matrix = torch.zeros((N, N))
        ngram_sets = [get_ngrams(q, n) for q in questions]
        for i in range(N):
            for j in range(i, N):
                set_i = ngram_sets[i]
                set_j = ngram_sets[j]
                union = set_i | set_j
                sim = 1.0 if not union else len(set_i & set_j) / len(union)
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
        return similarity_matrix

    # ======================================== #
    # ===Distribution Similarity Functions==== #
    # ======================================== #
    def compute_pairwise_cos_similarity(all_logits, use_softmax=True):
        device = all_logits.device
        N, D = all_logits.shape
        epsilon = 1e-10
        if use_softmax:
            all_probs = F.softmax(all_logits, dim=1)
        else:
            all_probs = all_logits / all_logits.sum(dim=1, keepdim=True)
            all_probs = torch.clamp(all_probs, min=epsilon)
        return torch.mm(F.normalize(all_probs, p=2, dim=1),
                        F.normalize(all_probs, p=2, dim=1).t())

    def compute_pairwise_kl_similarity(all_logits, epsilon=1e-10, use_softmax=True):
        device = all_logits.device
        N, D = all_logits.shape
        batch_size = 4
        if use_softmax:
            all_probs = F.softmax(all_logits, dim=1)
        else:
            all_probs = all_logits / all_logits.sum(dim=1, keepdim=True)
            all_probs = torch.clamp(all_probs, min=epsilon)

        all_probs = all_probs.to(dtype=torch.float32)
        log_all_probs = torch.log(all_probs + epsilon)
        similarity_matrix = torch.empty((N, N), device=device, dtype=torch.float32)
        num_batches = (N + batch_size - 1) // batch_size
        for i in range(num_batches):
            start_i = i * batch_size
            end_i = min((i + 1) * batch_size, N)
            P_batch = all_probs[start_i:end_i].unsqueeze(1)
            log_P_batch = log_all_probs[start_i:end_i].unsqueeze(1)
            Q = all_probs.unsqueeze(0)
            log_Q = log_all_probs.unsqueeze(0)
            kl_div = torch.sum(P_batch * (log_P_batch - log_Q), dim=2)
            similarity = 1 - kl_div
            similarity_matrix[start_i:end_i] = similarity
        return similarity_matrix

    def compute_pairwise_js_similarity(all_logits, epsilon=1e-10, use_softmax=True, batch_size=4):
        device = all_logits.device
        N, D = all_logits.shape
        if use_softmax:
            all_probs = F.softmax(all_logits, dim=1)
        else:
            all_probs = all_logits / all_logits.sum(dim=1, keepdim=True)
            all_probs = torch.clamp(all_probs, min=epsilon)
        all_probs = all_probs.to(dtype=torch.float32)
        similarity_matrix = torch.empty((N, N), device=device, dtype=torch.float32)
        num_batches = (N + batch_size - 1) // batch_size
        for i in range(num_batches):
            start_i = i * batch_size
            end_i = min((i + 1) * batch_size, N)
            P_batch = all_probs[start_i:end_i]
            P_exp = P_batch.unsqueeze(1)
            Q_exp = all_probs.unsqueeze(0)
            M = 0.5 * (P_exp + Q_exp)
            kl_pm = torch.sum(P_exp * torch.log((P_exp + epsilon) / (M + epsilon)), dim=2)
            kl_qm = torch.sum(Q_exp * torch.log((Q_exp + epsilon) / (M + epsilon)), dim=2)
            js_div = 0.5 * (kl_pm + kl_qm)
            js_similarity = 1 / (1 + js_div)
            similarity_matrix[start_i:end_i] = js_similarity
        return similarity_matrix

    # ======================================== #
    # =============LLM Functions============== #
    # ======================================== #
    def calculate_probability(prompt, tokenizer, model):
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        length = attention_mask[0].sum().item()
        log_probs = F.log_softmax(logits[0, :length-1, :], dim=-1)
        labels = input_ids[0, 1:length]
        token_log_probs = log_probs[range(length-1), labels]
        sequence_log_prob = token_log_probs.sum().item()
        return sequence_log_prob

    def calculate_probability_batch(prompts, tokenizer, model):
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        sequence_log_probs = []
        for i in range(len(prompts)):
            length = attention_mask[i].sum().item()
            log_probs = F.log_softmax(logits[i, :length-1, :], dim=-1)
            labels = input_ids[i, 1:length]
            token_log_probs = log_probs[range(length-1), labels]
            sequence_log_probs.append(token_log_probs.sum().item())
        return sequence_log_probs

    def get_last_hidden_state(model, tokenizer, prompt):
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1]
        averaged_hidden_state = last_hidden_state.mean(dim=1)
        return averaged_hidden_state.squeeze(0).tolist()

    def get_best_answer(prompt, answer_choices, tokenizer, model):
        prompts = [f"{prompt} {choice}" for choice in answer_choices]
        choice_probs = calculate_probability_batch(prompts, tokenizer, model)
        best_answer_index = torch.argmax(torch.tensor(choice_probs)).item()
        best_answer = answer_choices[best_answer_index]
        return best_answer, choice_probs

    def get_averaged_logits(prompt, answer, tokenizer, model):
        prompt_encoding = tokenizer(prompt, add_special_tokens=False)
        prompt_len = len(prompt_encoding["input_ids"])
        full_text = f"{prompt} {answer}"
        full_encoding = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            outputs = model(full_encoding["input_ids"])
            logits = outputs.logits
        answer_logits = logits[:, prompt_len:, :]
        averaged_logits = answer_logits.mean(dim=1)
        return averaged_logits.squeeze(0).tolist()

    # ======================================== #
    # ==========Filtering Functions=========== #
    # ======================================== #
    def get_indices_to_filter(similarity_matrix, N, k, tag, sanity_check, questions):
        k2name = {0.1:"f1", 0.25:"f2", 0.5:"f3", 0.75:"f4", 0.9:"f5"}
        name = k2name[k] + tag
        top_k_count = int(k * N)
        top_values, top_indices_flat = torch.topk(similarity_matrix.view(-1), N*N)
        top_indices = torch.unravel_index(top_indices_flat, similarity_matrix.shape)
        filtered_indices = set()
        result = []
        for i, j in tqdm(zip(top_indices[0].tolist(), top_indices[1].tolist()), disable=True):
            if i == j:
                continue
            if i not in filtered_indices and j not in filtered_indices:
                filtered_indices.add(j)
                result.append(j)
                sanity_check[questions[i]][name].append(questions[j])
            if len(result) >= top_k_count:
                break
        return result

    # ======================================== #
    # ===========Analysis Functions=========== #
    # ======================================== #
    def compute_avg_shannon_entropy(all_logits, filtered_indices):
        total_indices = set(range(all_logits.size(0)))
        remaining_indices = list(total_indices - set(filtered_indices))
        if len(remaining_indices) == 0:
            return 0.0
        remaining_logits = all_logits[remaining_indices]
        probs = F.softmax(remaining_logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs), dim=1)
        return entropy.mean().item()

    # ======================================== #
    # =======Load Model and Tokenizer========= #
    # ======================================== #
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    is_8bit = True
    models_to_quantize = ["facebook/opt-30b", "facebook/opt-13b", "meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.2-11B-Vision", "meta-llama/Llama-3.1-70B", "meta-llama/Llama-3.1-405B-FP8", "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-14B", "Qwen/Qwen2.5-32B", "Qwen/Qwen2.5-72B"]
    if model_name in models_to_quantize:
        model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=False)
        is_8bit = False
    st_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Process each dataset from the provided list
    for dataset_path in args.datasets:
        print("Processing dataset:", dataset_path)
        qa_dataset = QADataset(dataset_path.replace(".json", "").strip())
        qa_dataset.load_from_json(dataset_path)
        sanity_check = {}
        questions = qa_dataset.questions
        answer_choices_list = qa_dataset.answer_choices
        correct_indices = qa_dataset.correct_answer_indices
        question_embeddings = st_model.encode(questions)
        results = []
        all_logits_list = []

        for i in tqdm(range(len(qa_dataset)), desc=f"Processing QA Dataset {dataset_path}"):
            question = questions[i]
            answer_choices = answer_choices_list[i]
            correct_answer_idx = correct_indices[i]

            sanity_check[question] = {}
            for f in ["f1", "f2", "f3", "f4", "f5"]:
                for t in ["NS", "ES", "LS", "KL", "JS", "ELS", "EKL", "EJS"]:
                    sanity_check[question][f+t] = []

            best_answer, choice_probs = get_best_answer(question, answer_choices, tokenizer, model)
            try:
                predicted_index = answer_choices.index(best_answer)
                acc = 1 if predicted_index == correct_answer_idx else 0
            except ValueError:
                continue

            last_hidden_state = get_last_hidden_state(model, tokenizer, question)
            logits = get_averaged_logits(question, answer_choices[correct_answer_idx], tokenizer, model)
            logits_tensor = torch.tensor(logits)
            probs = F.softmax(logits_tensor, dim=0)
            topk = torch.topk(probs, 2)
            vocab_size = probs.size(0)
            shannon_entropy = -torch.sum(probs * torch.log(probs)).item()
            max_softmax_probability = torch.max(probs).item()
            prediction_margin = (topk.values[0] - topk.values[1]).item()
            perplexity = math.exp(shannon_entropy)
            logits_variance = torch.var(logits_tensor).item()
            kl_divergence_uniform = torch.sum(probs * torch.log(probs * vocab_size)).item()

            results.append({
                "question": question,
                "answer_choices": answer_choices,
                "choice_probs": choice_probs,
                "best_answer": best_answer,
                "acc": acc,
                "shannon_entropy": shannon_entropy,
                "max_softmax_probability": max_softmax_probability,
                "prediction_margin": prediction_margin,
                "perplexity": perplexity,
                "logits_variance": logits_variance,
                "kl_divergence_uniform": kl_divergence_uniform,
                "last_hidden_state": last_hidden_state,
            })
            all_logits_list.append(logits_tensor)

        if len(all_logits_list) == 0:
            continue
        all_logits = torch.stack(all_logits_list)

        question_ngram_matrix = compute_ngram_similarity(questions)
        question_similarity_matrix = compute_pairwise_cos_similarity(torch.tensor(question_embeddings))
        logits_similarity_matrix = compute_pairwise_cos_similarity(all_logits)
        logits_kl_matrix = compute_pairwise_kl_similarity(all_logits)
        logits_js_matrix = compute_pairwise_js_similarity(all_logits)

        combined_similarity_matrix = question_similarity_matrix * logits_similarity_matrix
        combined_kl_matrix = question_similarity_matrix * logits_kl_matrix
        combined_js_matrix = question_similarity_matrix * logits_js_matrix

        matrix_save_dir = "saved_matrices/"
        os.makedirs(matrix_save_dir, exist_ok=True)
        matrices_dict = {
            "question_ngram_matrix": question_ngram_matrix,
            "question_similarity_matrix": question_similarity_matrix,
            "logits_similarity_matrix": logits_similarity_matrix,
            "logits_kl_matrix": logits_kl_matrix,
            "logits_js_matrix": logits_js_matrix,
            "combined_similarity_matrix": combined_similarity_matrix,
            "combined_kl_matrix": combined_kl_matrix,
            "combined_js_matrix": combined_js_matrix
        }

        dataset_name = os.path.basename(dataset_path).replace(".json", "").strip()
        for name, matrix in matrices_dict.items():
            matrix_filename = f"{dataset_name}_{args.model_name.split('/')[-1].strip()}_{name}.npy"
            np.save(os.path.join(matrix_save_dir, matrix_filename), matrix.cpu().numpy())

        print("MODEL:", args.model_name)
        print("8 bit:", is_8bit)
        print("TASK:", dataset_path)
        names = ["NS", "ES", "LS", "KL", "JS", "ELS", "EKL", "EJS"]
        matrices = [
            question_ngram_matrix,
            question_similarity_matrix,
            logits_similarity_matrix,
            logits_kl_matrix,
            logits_js_matrix,
            combined_similarity_matrix,
            combined_kl_matrix,
            combined_js_matrix
        ]

        for matrix, tag in zip(matrices, names):
            idx_tr1 = get_indices_to_filter(matrix, len(results), 0.1, tag, sanity_check, questions)
            idx_tr2 = get_indices_to_filter(matrix, len(results), 0.25, tag, sanity_check, questions)
            idx_tr3 = get_indices_to_filter(matrix, len(results), 0.5, tag, sanity_check, questions)
            idx_tr4 = get_indices_to_filter(matrix, len(results), 0.75, tag, sanity_check, questions)
            idx_tr5 = get_indices_to_filter(matrix, len(results), 0.9, tag, sanity_check, questions)

            total_acc0 = sum([results[k]["acc"] for k in range(len(results))])
            total_acc1 = sum([results[k]["acc"] for k in range(len(results)) if k not in idx_tr1])
            total_acc2 = sum([results[k]["acc"] for k in range(len(results)) if k not in idx_tr2])
            total_acc3 = sum([results[k]["acc"] for k in range(len(results)) if k not in idx_tr3])
            total_acc4 = sum([results[k]["acc"] for k in range(len(results)) if k not in idx_tr4])
            total_acc5 = sum([results[k]["acc"] for k in range(len(results)) if k not in idx_tr5])

            avg_ent0 = compute_avg_shannon_entropy(all_logits, [])
            avg_ent1 = compute_avg_shannon_entropy(all_logits, idx_tr1)
            avg_ent2 = compute_avg_shannon_entropy(all_logits, idx_tr2)
            avg_ent3 = compute_avg_shannon_entropy(all_logits, idx_tr3)
            avg_ent4 = compute_avg_shannon_entropy(all_logits, idx_tr4)
            avg_ent5 = compute_avg_shannon_entropy(all_logits, idx_tr5)

            print("Filtered Accuracy {}:".format(tag) +
                  str(float(total_acc0/len(results))) + " " +
                  str(float(total_acc1/(len(results)-len(idx_tr1)))) + " " +
                  str(float(total_acc2/(len(results)-len(idx_tr2)))) + " " +
                  str(float(total_acc3/(len(results)-len(idx_tr3)))) + " " +
                  str(float(total_acc4/(len(results)-len(idx_tr4)))) + " " +
                  str(float(total_acc5/(len(results)-len(idx_tr5)))))

            print("Average Shannon Entropy {}:".format(tag),
                  avg_ent0, avg_ent1, avg_ent2, avg_ent3, avg_ent4, avg_ent5)

            counter = 0
            logits_to_vis = [all_logits[i] for i in range(len(results))]
            logits_to_vis_f1 = [all_logits[i] for i in range(len(results)) if i not in idx_tr1]
            logits_to_vis_f2 = [all_logits[i] for i in range(len(results)) if i not in idx_tr2]
            logits_to_vis_f3 = [all_logits[i] for i in range(len(results)) if i not in idx_tr3]
            logits_to_vis_f4 = [all_logits[i] for i in range(len(results)) if i not in idx_tr4]
            logits_to_vis_f5 = [all_logits[i] for i in range(len(results)) if i not in idx_tr5]

            for logits_group in [logits_to_vis, logits_to_vis_f1, logits_to_vis_f2, logits_to_vis_f3, logits_to_vis_f4, logits_to_vis_f5]:
                try:
                    if len(logits_group) > 30:
                        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
                    else:
                        tsne = TSNE(n_components=2, random_state=42, perplexity=len(logits_group), n_iter=1000)
                    log = torch.tensor(np.array(logits_group)).cpu()
                    data_2d = tsne.fit_transform(log)
                    k = 5
                    n_clusters = int(len(log) / k) + 1
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    kmeans.fit(data_2d.astype(np.float64))
                    labels = kmeans.labels_
                    cluster_indices = {}
                    colors_by_acc = []
                    for index, label in enumerate(labels):
                        color = 'blue' if results[index]["acc"] == 1 else 'red'
                        colors_by_acc.append(color)
                        if label not in cluster_indices:
                            cluster_indices[float(label)] = []
                        cluster_indices[float(label)].append({
                            "question": questions[index],
                            "coordinates": [float(coord) for coord in data_2d[index]],
                            "acc": results[index]["acc"]
                        })
                    x_min, x_max = data_2d[:, 0].min() - 1, data_2d[:, 0].max() + 1
                    y_min, y_max = data_2d[:, 1].min() - 1, data_2d[:, 1].max() + 1
                    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
                    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()].astype(np.float64))
                    Z = Z.reshape(xx.shape)
                    plt.figure(figsize=(10, 8))
                    plt.contourf(xx, yy, Z, alpha=0.5, cmap='viridis')
                    scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=colors_by_acc, edgecolor=colors_by_acc, cmap='viridis', s=5)
                    plt.title('{} t-SNE with k-Means Clustering, Level = {}'.format(dataset_path, counter))
                    plt.xlabel('t-SNE Component 1')
                    plt.ylabel('t-SNE Component 2')
                    plt.colorbar(scatter, ticks=range(n_clusters), label='Cluster Label')
                    save_dir = os.path.dirname("visualization/"+dataset_name+"/"+tag+"/"+args.model_name.split("/")[-1]+"/"+str(counter)+"clustering_{}.png".format(tag))
                    os.makedirs(save_dir, exist_ok=True)
                    plt.savefig(os.path.join(save_dir, str(counter)+"clustering_{}.png".format(tag)),
                                dpi=300, bbox_inches='tight')
                    json_dir = os.path.dirname("clustering/"+dataset_name+"/"+tag+"/"+args.model_name.split("/")[-1]+"/"+str(counter)+"clustering_{}.json".format(tag))
                    os.makedirs(json_dir, exist_ok=True)
                    with open(os.path.join(json_dir, str(counter)+"clustering_{}.json".format(tag)), "w") as f:
                        json.dump(cluster_indices, f, ensure_ascii=False, indent=2)
                    counter += 1
                except ValueError:
                    counter += 1

        output_filename = f"{dataset_name}_{args.model_name.replace('/', '_')}.json"
        with open(output_filename, "w") as f:
            json.dump(results, f, indent=2)
        with open(output_filename.replace(".json", "")+"_sanity_check.json", "w") as f:
            json.dump(sanity_check, f, indent=2)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process multiple QA datasets with a single model")
    parser.add_argument('--model_name', type=str, required=True, help="Name of the pretrained model")
    parser.add_argument('--datasets', type=str, nargs='+', required=True, help="List of dataset JSON files")
    return parser.parse_args()

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    args = parse_arguments()
    set_seed(42)
    main(args)