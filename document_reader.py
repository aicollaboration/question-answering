
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from collections import OrderedDict
import torch


class DocumentReader:
    def __init__(self, pretrained_model_name_or_path):
        self.reader_path = pretrained_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.reader_path)
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            self.reader_path)
        self.max_len = self.model.config.max_position_embeddings
        self.chunked = False

    def chunkify(self):
        qmask = self.inputs['token_type_ids'].lt(1)
        qt = torch.masked_select(self.inputs['input_ids'], qmask)
        chunk_size = self.max_len - qt.size()[0] - 1  # the "-1" accounts for

        chunked_input = OrderedDict()
        for k, v in self.inputs.items():
            q = torch.masked_select(v, qmask)
            c = torch.masked_select(v, ~qmask)
            chunks = torch.split(c, chunk_size)

            for i, chunk in enumerate(chunks):
                if i not in chunked_input:
                    chunked_input[i] = {}

                thing = torch.cat((q, chunk))
                if i != len(chunks) - 1:
                    if k == 'input_ids':
                        thing = torch.cat((thing, torch.tensor([102])))
                    else:
                        thing = torch.cat((thing, torch.tensor([1])))

                chunked_input[i][k] = torch.unsqueeze(thing, dim=0)
        return chunked_input

    def ask(self, question, text):
        self.inputs = self.tokenizer.encode_plus(
            question, text, add_special_tokens=True, return_tensors="pt")
        self.input_ids = self.inputs.input_ids.tolist()[0]

        if len(self.input_ids) > self.max_len:
            self.inputs = self.chunkify()
            self.chunked = True

        if self.chunked:
            answer = ''
            for k, chunk in self.inputs.items():
                answer_start_scores, answer_end_scores = self.model(**chunk)

                answer_start = torch.argmax(answer_start_scores)
                answer_end = torch.argmax(answer_end_scores) + 1

                ans = self.convert_ids_to_string(chunk['input_ids'][0][answer_start:answer_end])
                if ans != '[CLS]':
                    answer += ans + " / "
            return answer
        else:
            answer_start_scores, answer_end_scores = self.model(**self.inputs)
            answer_start = torch.argmax(answer_start_scores)
            answer_end = torch.argmax(answer_end_scores) + 1

            return self.convert_ids_to_string(self.inputs.input_ids[0][answer_start:answer_end])

    def convert_ids_to_string(self, input_ids):
        return self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_ids))
