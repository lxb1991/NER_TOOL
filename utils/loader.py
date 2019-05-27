from os import path


def load_data_by_lm_config(lm_config):
    with open(lm_config) as fi:
        file_dict = {}
        for line_text in fi.readlines():
            if line_text.startswith("#"):
                continue
            pairs = line_text.strip().split("=")
            assert len(pairs) == 2, "please confirm the format of lm config file"
            file_dict[pairs[0]] = pairs[1]
        return file_dict


def load_data_by_base_config(base_config):
    with open(base_config) as fi:
        file_dict = {}
        for line_text in fi.readlines():
            if line_text.startswith("#"):
                continue
            pairs = line_text.strip().split("=")
            assert len(pairs) == 2, "please confirm the format of base config file"
            file_dict[pairs[0]] = pairs[1]
        return file_dict


def load_data_by_config(config_path):
    with open(path.join(config_path, 'config')) as fi:
        file_dict = {}
        for line_text in fi.readlines():
            if line_text.startswith("#"):
                continue
            pairs = line_text.strip().split("=")
            assert len(pairs) == 2, "please confirm the format of base config file"
            file_dict[pairs[0]] = path.join(config_path, pairs[1])
        return file_dict


def load_data_by_cross_config(cross_config):
    model_matrix = {}
    model_tasks = {}
    model_domains = {}
    model_coefficient = {}
    with open(cross_config) as fi:
        for lines in fi.readlines():
            if lines.startswith("#"):
                continue
            pairs = lines.strip().split("=")
            model_type = pairs[0]
            model_path = pairs[1]
            model_coef = pairs[2]
            t, d = model_type.split("_")
            if t not in model_tasks:
                model_tasks[t] = len(model_tasks)
            if d not in model_domains:
                model_domains[d] = len(model_domains)
            model_matrix[(t, d)] = model_path
            model_coefficient[(t, d)] = model_coef
    return model_matrix, model_tasks, model_domains, model_coefficient


def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word


def load_conll_data_by_files(file_path):
    conll_data = []
    conll_sentence = []
    with open(file_path) as fi:
        for lines in fi.readlines():
            line_text = lines.strip()
            if line_text:
                pairs = line_text.split()
                word = normalize_word(pairs[0])
                tag = pairs[1]
                conll_sentence.append((word, tag))
            else:
                if len(conll_sentence) > 0:
                    conll_data.append(conll_sentence)
                conll_sentence = []
    return conll_data


def load_lm_data_by_files(file_path):
    lm_data = []
    with open(file_path) as fi:
        for lines in fi.readlines():
            lm_sentence = []
            line_text = lines.strip()
            if line_text:
                for word in line_text.split():
                    lm_sentence.append(word)
            if len(lm_sentence) > 0:
                lm_data.append(lm_sentence)
    return lm_data


def load_word_from_conll(file_path):
    word_data = set()
    with open(file_path) as fi:
        for lines in fi.readlines():
            line_text = lines.strip()
            if line_text:
                pairs = line_text.split()
                word = normalize_word(pairs[0])
                word_data.add(word)
    return word_data


def load_word_from_lm(file_path):
    word_data = set()
    with open(file_path) as fi:
        for lines in fi.readlines():
            for text in lines.strip().split(" "):
                if text.strip():
                    word_data.add(normalize_word(text.strip()))
    return word_data


def load_label_from_conll(file_path):
    label_data = set()
    with open(file_path) as fi:
        for lines in fi.readlines():
            line_text = lines.strip()
            if line_text:
                pairs = line_text.split()
                label = pairs[1]
                label_data.add(label)
    return label_data
