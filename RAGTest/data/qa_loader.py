import json
import os
import re
from pathlib import Path

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

DATA_DIR = Path(__file__).resolve().parent


def build_split(answers, questions, supporting_facts, title2id, title2sentences):
    golden_ids = []
    golden_sentences = []
    filter_questions = []
    filter_answers = []
    i = 0
    for sup, q, a in zip(supporting_facts, questions, answers):
        # i = i + 1
        # if i == 300:
        #     break
        # if len(sup['sent_id']) == 0:
        #     continue
        try:
            sup_title = sup['title']
            # send_id = sup['sent_id']
            # golden_id = [title2start[t]+i for i,t in zip(send_id,sup_title)]
            sup_titles = set(sup_title)
            golden_id = [title2id[t] for t in sup_titles]


        except:
            continue
        golden_ids.append(golden_id)
        golden_sentences.append([' '.join(title2sentences[t]) for t in sup_titles])
        filter_questions.append(q)
        filter_answers.append(a)
    print("questions:", len(questions))
    print("filter_questions:", len(filter_questions))
    return filter_questions,filter_answers, golden_ids, golden_sentences

def extract_law_name(input_str):
    # 使用正则表达式匹配第一个 '-' 字符后的非汉字字符串
    match = re.search(r'-(.*?)[^\u4e00-\u9fa5]', input_str)

    if match:
        law_name = match.group(1).strip('-')
        return law_name
    else:
        return None


def _hotpot_split_name() -> str:
    return os.environ.get("HOTPOT_SPLIT", "combined").strip().lower()


def _hotpot_max_examples() -> int:
    try:
        return max(int(os.environ.get("HOTPOT_MAX_EXAMPLES", "300")), 0)
    except ValueError:
        return 300


def _iter_hotpot_records(dataset, split_name: str):
    if split_name == "combined":
        split_order = ("train", "validation", "test")
    else:
        split_order = (split_name,)
    for split in split_order:
        for row in dataset[split]:
            yield row


def _build_hotpot_dataset(dataset, split_name: str, max_examples: int):
    title2sentences = {}
    titles = []
    title2start = {}
    title2id = {}
    source_sentences = []
    golden_ids = []
    golden_sentences = []
    filter_questions = []
    filter_answers = []
    example_count = 0
    title_cursor = 0

    for row in _iter_hotpot_records(dataset, split_name):
        if max_examples and example_count >= max_examples:
            break
        supporting = row.get("supporting_facts", {})
        sup_titles = supporting.get("title", []) or []
        if not sup_titles:
            continue
        context = row.get("context", {})
        context_titles = context.get("title", []) or []
        context_sentences = context.get("sentences", []) or []
        local_title2sentences = {
            title: sentences for title, sentences in zip(context_titles, context_sentences)
        }

        unique_titles = []
        seen = set()
        for title in sup_titles:
            if title in seen or title not in local_title2sentences:
                continue
            seen.add(title)
            unique_titles.append(title)
        if not unique_titles:
            continue

        for title in context_titles:
            if title not in title2sentences:
                sentences = local_title2sentences.get(title, [])
                title2sentences[title] = sentences
                title2start[title] = title_cursor
                titles.append(title)
                source_sentences.extend(sentences)
                title_cursor += len(sentences)
                title2id[title] = len(title2id)

        golden_ids.append([title2id[title] for title in unique_titles])
        golden_sentences.append([" ".join(local_title2sentences[title]) for title in unique_titles])
        filter_questions.append(row.get("question", ""))
        filter_answers.append(row.get("answer", ""))
        example_count += 1

    print("hotpot_split:", split_name)
    print("hotpot_examples:", example_count)
    return dict(
        question=filter_questions,
        answers=filter_answers,
        golden_ids=golden_ids,
        golden_sentences=golden_sentences,
        sources=source_sentences,
        titles=titles,
        title2sentences=title2sentences,
        title2start=title2start,
        title2id=title2id,
        dataset=dataset,
    )


def get_qa_dataset(dataset_name: str):
    if dataset_name == "hotpot_qa":
        from datasets import load_dataset

        dataset = load_dataset("hotpot_qa", "fullwiki")
        return _build_hotpot_dataset(dataset, _hotpot_split_name(), _hotpot_max_examples())


    elif dataset_name == "json_download":
        # with open("data/data_100.json", 'r', encoding='utf-8') as file:
        with (DATA_DIR / "data_50.json").open('r', encoding='utf-8') as file:
            data = json.load(file)
        questions = []
        answers = []
        golden_sources = []
        golden_ids = []
        question_types = []
        text2id = {}
        dataset = data
        for entry in data:
            question_types.append(entry["other_info"]["question_type"])
            entry = entry["key_content"]
            question = entry['question']
            answer = entry['answer']
            references = entry['reference']
            ids = entry["reference_idx"]
            if answer=="":
                continue
            questions.append(question)
            answers.append(answer)
            golden_sources.append(references)
            golden_ids.append(ids)
        
    #     for entry in data:
    #         title = entry["other_info"]["doc_name"]
    #         for reference, ids in zip(entry["key_content"]["reference"], entry["key_content"]["reference_idx"]):
    #             text = reference
    #             id = ids
    #             ducument = Document(text=text, metadata={'title': title, 'id': id}, doc_id=str(id))
    #             documents.append(ducument)
        
    #     with open("/root/autodl-tmp/zh/ragx_old/data/qa.json", 'r', encoding='utf-8') as file:
    #         data = json.load(file)

    #     questions = []
    #     answers = []
    #     golden_sources = []
    #     golden_ids = []
    #     text2id = {}

    #     dataset = data

    #     for entry in data:
    #         question = entry['question']
    #         answer = entry['answer']
    #         references = entry['reference']
    #         ids = entry['ids']
    #         if answer=="":
    #             continue
    #         questions.append(question)
    #         answers.append(answer)
    #         golden_sources.append(references)
    #         golden_ids.append(ids)

    # else:
    #     raise NotImplementedError(f'dataset {dataset_name} not implemented!')

    return dict(
        question=questions,
        answers=answers,
        golden_sentences=golden_sources,
        golden_ids=golden_ids,
        question_types=question_types,
        dataset=dataset,
        title2id=text2id)


if __name__ == '__main__':
    name = 'json_download'  # drop, natural_questions, hotpot_qa
    data = get_qa_dataset(name)
