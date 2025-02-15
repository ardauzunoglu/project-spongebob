import json
from datasets import load_dataset
from torch.utils.data import Dataset

class QADataset(Dataset):
    """
    A PyTorch Dataset to hold questions and indices of the correct answer choices.
    """
    def __init__(self, dataset_name):
        """
        Args:
            hf_dataset: A loaded Hugging Face Dataset object or any list-like structure
                        where each element has 'question_stem' and 'answerKey'.
        """
        self.dataset = dataset_name
        self.questions = []
        self.answer_choices = []
        self.correct_answer_indices = []
        self.perturbation_rate = 0

    def __len__(self):
        """
        Returns the total number of examples in the dataset.
        """
        return len(self.questions)

    def __getitem__(self, idx):
        """
        Retrieves the (question, correct_answer_index) pair for the given index.
        """
        return {
            "question": self.questions[idx],
            "answer_choices": self.answer_choices[idx],
            "correct_answer_index": self.correct_answer_indices[idx]
        }

    def fill(self, hf_dataset):
        ds2qf = {"openbookqa":"question_stem",
                 "arc_challenge":"question",
                 "arc_easy":"question",
                 "copa":["premise", "question"],
                 "commonsense_qa": "question",
                 "logiqa": ["context", "question"],
                 "piqa": "goal",
                 "pubmed_qa": "QUESTION",
                 "social_i_qa": ["context", "question"],
                 "truthful_qa": "question",
                 "winogrande": "sentence",
                 "boolq": ["passage", "question"]
                 } # Convert dataset name to question stem

        for example in hf_dataset:
            question_field = ds2qf[self.dataset]
            if type(question_field) == list:
                question_text = f"{example[question_field[0]]} {example[question_field[1]]}"
            else:
                question_text = example[question_field]

            if self.dataset == "openbookqa":
                answer_choices = example["choices"]["text"]
                try:
                    answer_key = ["A", "B", "C", "D"].index(example["answerKey"])
                except ValueError:
                    continue

            elif (self.dataset == "arc_challenge") or (self.dataset == "arc_easy"):
                answer_choices = example["choices"]["text"]
                answer_key = example["choices"]["label"].index(example["answerKey"])

            elif self.dataset == "copa":
                answer_choices = [example["choice1"], example["choice2"]]
                answer_key = example["label"]

            elif self.dataset == "commonsense_qa":
                answer_choices = example["choices"]["text"]
                answer_key = {"A":0,"B":1,"C":2,"D":3,"E":4}[example["answerKey"]]

            elif self.dataset == "logiqa":
                answer_choices = example["options"]  # Should be a list of option texts
                answer_key = {"a":0, "b":1, "c":2, "d":3, "e":4}[example["label"]]

            elif self.dataset == "piqa":
                answer_choices = [example["sol1"], example["sol2"]]
                answer_key = example["label"]

            elif self.dataset == "pubmed_qa":
                correct_answer_str = example["final_decision"]
                answer_choices = ["yes", "no", "maybe"]
                answer_mapping = {"yes": 0, "no": 1, "maybe": 2}
                answer_key = answer_mapping.get(correct_answer_str, -1)

            elif self.dataset == "social_i_qa":
                answer_choices = [example["answerA"], example["answerB"], example["answerC"]]  # Adjust based on dataset structure
                answer_key = int(example["label"]) - 1

            elif self.dataset == "truthful_qa":
                answer_choices = example["mc1_targets"]["choices"]   # Should be a list of option texts
                answer_key = example["mc1_targets"]["labels"].index(1)

            elif self.dataset == "winogrande":
                answer_choices = [example["option1"], example["option2"]]
                answer = example["answer"]
                if isinstance(answer, str):
                    if answer.lower() in ['option1', '1']:
                        answer_key = 0
                    elif answer.lower() in ['option2', '2']:
                        answer_key = 1
                    else:
                        print(f"Unknown answer format: {answer}")
                        continue
                elif isinstance(answer, int):
                    answer_key = answer - 1

            elif self.dataset == "boolq":
                answer_choices = ["True", "False"]
                answer_key = 0 if bool(example["answer"]) else 1

            self.questions.append(question_text)
            self.answer_choices.append(answer_choices)
            self.correct_answer_indices.append(answer_key)

    def save_to_json(self, save_path):
        to_save = {
                    "questions":self.questions,
                    "answer_choices":self.answer_choices,
                    "correct_answer_indices":self.correct_answer_indices,
                    "perturbation_rate":self.perturbation_rate
                    }

        with open(save_path, "w") as file:
            json.dump(to_save, file, indent=4)

    def load_from_json(self, path):
        with open(path, "r") as file:
            dataset = json.load(file)
            self.questions = dataset["questions"]
            self.answer_choices = dataset["answer_choices"]
            self.correct_answer_indices = dataset["correct_answer_indices"]
            self.perturbation_rate = dataset["perturbation_rate"]