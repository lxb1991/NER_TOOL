import argparse
from utils import loader, conlleval
from utils.conll_data import WordVocab, LabelVocab, BaseData, LMData
from torch.utils.data import DataLoader
from model.base import BaseNER, BaseLM
from model.cross import CrossNER
from torch import optim
from torch.nn import CrossEntropyLoss
from layers.sampled_softmax_loss import SampledSoftmaxLoss
import torch
import os
import logging
import math


def parse_args():
    parser = argparse.ArgumentParser(description="ner network")
    parser.add_argument("--train", action="store_true", help="train the model")
    parser.add_argument("--evaluate", action="store_true", help="evaluate the model on dev set")
    parser.add_argument("--predict", action="store_true", help="predict the label on test set with trained model")
    parser.add_argument("--mode", type=str, default="base")
    parser.add_argument("--gpu", type=str, default="0", help="specify gpu device")

    model_settings = parser.add_argument_group("single ner model settings")
    model_settings.add_argument("--word_embed_dim", type=int, default=100)
    model_settings.add_argument("--char_embed_dim", type=int, default=30)

    model_settings.add_argument("--lstm_hidden", type=int, default=100)
    model_settings.add_argument("--lstm_layer_num", type=int, default=1)
    model_settings.add_argument("--char_cnn_kernels", type=int, default=30)
    model_settings.add_argument("--crf_flag", type=bool, default=False)

    train_settings = parser.add_argument_group("train settings")
    train_settings.add_argument("--optimizer", type=str, default="SGD")
    train_settings.add_argument("--batch_size", type=int, default=10)
    train_settings.add_argument("--drop_out", type=float, default=0.5)
    train_settings.add_argument("--epochs", type=int, default=100)
    train_settings.add_argument("--lr", type=float, default=0.015)
    train_settings.add_argument("--lr_decay", type=float, default=0.05)
    train_settings.add_argument("--momentum", type=float, default=0)
    train_settings.add_argument("--l2", type=float, default=1e-8)

    path_settings = parser.add_argument_group("path settings")
    path_settings.add_argument("--base_config", type=str, default="data/base_config")
    path_settings.add_argument("--cross_config", type=str, default="data/multi_config")
    path_settings.add_argument("--lm_config", type=str, default="data/lm_config")
    path_settings.add_argument("--vocab_file", type=str, default="vocab/glove.6B.100d.txt")
    path_settings.add_argument("--log_path", type=str)
    args = parser.parse_args()
    return args


def optimizer(args, parameters):
    model_optimizer = None
    if args.optimizer == "SGD":
        model_optimizer = optim.SGD(parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.l2)
    elif args.optimizer == "ADA_GRAD":
        model_optimizer = optim.Adagrad(parameters, lr=args.lr, weight_decay=args.l2)
    elif args.optimizer == "ADA_DELTA":
        model_optimizer = optim.Adadelta(parameters, lr=args.lr, weight_decay=args.l2)
    elif args.optimizer == "RMS_PROP":
        model_optimizer = optim.RMSprop(parameters, lr=args.lr, weight_decay=args.l2)
    elif args.optimizer == "ADAM":
        model_optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.l2)
    else:
        print("Optimizer illegal: %s" % args.optimizer)
        exit(1)
    return model_optimizer


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr / (1 + decay_rate * epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def run_lm_model(hyper_param, device):
    logger = logging.getLogger("ner")
    logger.info("run lm model ...")

    files_dict = loader.load_data_by_lm_config(hyper_param.lm_config)
    word_vocab = WordVocab(hyper_param.word_embed_dim, hyper_param.char_embed_dim)
    word_vocab.load_lm_data_vocab(files_dict)
    word_vocab.load_word_embeddings(hyper_param.vocab_file)
    word_vocab.random_char_embeddings()
    logger.info("load word vocab down {}".format(len(word_vocab)))
    train_data_set_dir = files_dict['train']
    train_data_set = LMData(train_data_set_dir, word_vocab)
    train_iter = DataLoader(dataset=train_data_set, batch_size=hyper_param.batch_size, shuffle=True,
                            num_workers=1, collate_fn=train_data_set.padding)

    base_lm_model = BaseLM(hyper_param, word_vocab.word_embeddings, word_vocab.char_embeddings,
                           len(word_vocab), device)

    model_optimizer = optimizer(hyper_param, base_lm_model.parameters())

    criterion = SampledSoftmaxLoss(len(word_vocab), hyper_param.lstm_hidden, 100, gpu=True if device == 'cuda' else False,
                                   sparse=False, unk_id=1)
    if device == 'cuda':
        base_lm_model = base_lm_model.cuda()
        criterion = criterion.cuda()

    for epoch_idx in range(hyper_param.epochs):
        logger.info("train epoch start: {}".format(epoch_idx))

        base_lm_model.train()
        total_loss = 0
        batch_num = 0
        for train_batch in train_iter:
            tokens_idx, chars_idx, forward_idx, backward_idx, tokens_mask, tokens_len = train_batch
            final_forward, final_backward, forward_idx, backward_idx, word_len = \
                base_lm_model(tokens_idx, chars_idx, forward_idx, backward_idx, tokens_mask, tokens_len)

            loss = criterion(final_forward.view(-1, hyper_param.lstm_hidden), forward_idx.view(-1)) +\
                   criterion(final_backward.view(-1, hyper_param.lstm_hidden), backward_idx.view(-1))

            loss = loss / 2 / word_len.sum(0)
            total_loss += loss.item()
            loss.backward()
            model_optimizer.step()
            base_lm_model.zero_grad()
            batch_num += 1
        logger.info("train down, the lm perplexity: {0} ".format(math.exp(total_loss/batch_num)))


def run_base_ner_model(hyper_param, device):
    logger = logging.getLogger("ner")
    logger.info("run base ner model ...")

    files_dict = loader.load_data_by_base_config(hyper_param.base_config)
    word_vocab = WordVocab(hyper_param.word_embed_dim, hyper_param.char_embed_dim)
    word_vocab.load_conll_word_vocab(files_dict)
    word_vocab.load_word_embeddings(hyper_param.vocab_file)
    word_vocab.random_char_embeddings()
    logger.info("load word vocab down {}".format(len(word_vocab)))

    label_vocab = LabelVocab()
    label_vocab.load_label_vocab(files_dict)
    logger.info("load label vocab down {}".format(len(label_vocab)))

    train_data_set_dir = files_dict['train']
    train_data_set = BaseData(train_data_set_dir, word_vocab, label_vocab)
    train_iter = DataLoader(dataset=train_data_set, batch_size=hyper_param.batch_size, shuffle=True,
                            num_workers=1, collate_fn=train_data_set.padding)

    dev_data_set_dir = files_dict['dev']
    dev_data_set = BaseData(dev_data_set_dir, word_vocab, label_vocab)
    dev_iter = DataLoader(dataset=dev_data_set, batch_size=hyper_param.batch_size, shuffle=False,
                          num_workers=1, collate_fn=train_data_set.padding)

    test_data_set_dir = files_dict['test']
    test_data_set = BaseData(test_data_set_dir, word_vocab, label_vocab)
    test_iter = DataLoader(dataset=test_data_set, batch_size=hyper_param.batch_size, shuffle=False,
                           num_workers=1, collate_fn=train_data_set.padding)

    logger.info("train num {0} dev num {1} test num {2}".format(len(train_data_set), len(dev_data_set),
                                                                len(test_data_set)))

    base_ner_model = BaseNER(hyper_param, word_vocab.word_embeddings, word_vocab.char_embeddings,
                             len(label_vocab), device)
    if device == 'cuda':
        base_ner_model = base_ner_model.cuda()

    model_optimizer = optimizer(hyper_param, base_ner_model.parameters())
    criterion = CrossEntropyLoss(ignore_index=0, reduction='sum')
    for epoch_idx in range(hyper_param.epochs):
        logger.info("train epoch start: {}" .format(epoch_idx))
        if hyper_param.optimizer == "SGD":
            model_optimizer = lr_decay(model_optimizer, epoch_idx, hyper_param.lr_decay, hyper_param.lr)
        base_ner_model.train()
        total_loss = 0
        batch_num = 0
        for train_batch in train_iter:
            tokens_idx, chars_idx, tags_idx, _, sentence_len, _, _ = train_batch
            token_score, _, tags_idx = base_ner_model(tokens_idx, chars_idx, sentence_len, tags_idx)
            loss = criterion(token_score.view(-1, token_score.shape[-1]), tags_idx.view(-1))
            loss = loss / tokens_idx.size(0)
            total_loss += loss.item()
            loss.backward()
            model_optimizer.step()
            base_ner_model.zero_grad()
            batch_num += 1
        logger.info("train down total loss {0} for batch num {1}".format(total_loss, batch_num))

        base_ner_model.eval()
        evaluation(dev_iter, base_ner_model, label_vocab, logger)
        evaluation(test_iter, base_ner_model, label_vocab, logger)


def evaluation(dev_iter, model, label_vocab, logger):
    golden_label = []
    predict_label = []
    with torch.no_grad():
        for dev_batch in dev_iter:
            tokens_idx, chars_idx, tags_idx, tokens_mask, sentence_len, _, _ = dev_batch
            token_score, token_hat, tags_idx = model(tokens_idx, chars_idx, sentence_len, tags_idx)
            for p_seq, g_seq, t_seq in zip(token_hat.cpu().numpy().tolist(), tags_idx.cpu().numpy().tolist(),
                                           tokens_mask.cpu().numpy().tolist()):
                g_labels = []
                p_labels = []
                for p_label_idx, g_label_idx, t_mask in zip(p_seq, g_seq, t_seq):
                    if t_mask == 1:
                        p_label = label_vocab.get_label(p_label_idx)
                        g_label = label_vocab.get_label(g_label_idx)
                        p_labels.append(p_label)
                        g_labels.append(g_label)
                golden_label.extend(g_labels)
                predict_label.extend(p_labels)
    precision, recall, f1 = conlleval.evaluate(golden_label, predict_label, verbose=False)
    logger.info("eval precision {0}, recall {1}, f1 {2}.".format(precision, recall, f1))
    return f1


def run_cross_ner_model(hyper_param, device):
    logger = logging.getLogger("ner")
    logger.info("run cross ner model ...")

    data_matrix_dir, task_vocab, domain_vocab, coefficient = loader.load_data_by_cross_config(hyper_param.cross_config)
    logger.info("data matrix dir {}".format(data_matrix_dir))
    logger.info("task_vocab {}".format(task_vocab))
    logger.info("domain_vocab {}".format(domain_vocab))

    data_matrix_vocab = {}
    for key in data_matrix_dir:
        data_matrix_vocab[key] = len(data_matrix_vocab)

    multi_train_iter = {}
    multi_dev_iter = {}
    multi_test_iter = {}
    multi_label_vocab = {}

    lm_matrix_dir = {key: value for key, value in data_matrix_dir.items() if key[0] == 'lm'}
    ner_matrix_dir = {key: value for key, value in data_matrix_dir.items() if key[0] == 'ner'}
    word_vocab = WordVocab(hyper_param.word_embed_dim, hyper_param.char_embed_dim)
    for config_path in ner_matrix_dir.values():
        word_vocab.load_conll_word_vocab(loader.load_data_by_config(config_path))
    word_vocab.load_lm_data_vocab(lm_matrix_dir)
    word_vocab.load_word_embeddings(hyper_param.vocab_file)
    word_vocab.random_char_embeddings()
    logger.info("load word vocab down {}".format(len(word_vocab)))

    for item_key, item_dir in ner_matrix_dir.items():
        files_dict = loader.load_data_by_config(item_dir)

        label_vocab = LabelVocab()
        label_vocab.load_label_vocab_by_files(files_dict['train'])
        label_vocab.load_label_vocab_by_files(files_dict['dev'])
        label_vocab.load_label_vocab_by_files(files_dict['test'])
        logger.info("load label vocab down {}".format(len(label_vocab)))
        multi_label_vocab[item_key] = label_vocab

        multi_train_data_set = BaseData(files_dict['train'], word_vocab, label_vocab)
        train_iter = DataLoader(dataset=multi_train_data_set, batch_size=hyper_param.batch_size, shuffle=True,
                                num_workers=1, collate_fn=multi_train_data_set.padding)
        multi_train_iter[item_key] = train_iter

        multi_dev_data_set = BaseData(files_dict['dev'], word_vocab, label_vocab)
        dev_iter = DataLoader(dataset=multi_dev_data_set, batch_size=hyper_param.batch_size, shuffle=False,
                              num_workers=1, collate_fn=multi_dev_data_set.padding)
        multi_dev_iter[item_key] = dev_iter

        multi_test_data_set = BaseData(files_dict['test'], word_vocab, label_vocab)
        test_iter = DataLoader(dataset=multi_test_data_set, batch_size=hyper_param.batch_size, shuffle=False,
                               num_workers=1, collate_fn=multi_test_data_set.padding)
        multi_test_iter[item_key] = test_iter

        logger.info("load ner data {0} down, train size {1}, dev size {2}, test size {3}".
                    format(item_key, len(multi_train_data_set), len(multi_dev_data_set), len(multi_test_data_set)))

    for item_key, item_dir in lm_matrix_dir.items():
        multi_train_data_set = LMData(item_dir, word_vocab)
        train_iter = DataLoader(dataset=multi_train_data_set, batch_size=hyper_param.batch_size, shuffle=True,
                                num_workers=1, collate_fn=multi_train_data_set.padding)
        multi_train_iter[item_key] = train_iter

        logger.info("load lm data{0}, train {1}".format(item_key, len(multi_train_iter[item_key])))

    cross_ner_model = CrossNER(hyper_param, word_vocab.word_embeddings, word_vocab.char_embeddings,
                               multi_label_vocab, len(task_vocab), len(domain_vocab), ner_matrix_dir, device)

    model_optimizer = optimizer(hyper_param, cross_ner_model.parameters())
    ner_criterion = CrossEntropyLoss(ignore_index=0, reduction='mean')
    lm_criterion = SampledSoftmaxLoss(len(word_vocab), hyper_param.lstm_hidden, 100,
                                      gpu=True if device == 'cuda' else False, sparse=False, unk_id=1)

    if device == 'cuda':
        cross_ner_model = cross_ner_model.cuda()
        lm_criterion = lm_criterion.cuda()

    for epoch_idx in range(hyper_param.epochs):
        print("train epoch start: {}" .format(epoch_idx))
        cross_ner_model.train()

        multi_train_flag = {}
        multi_train_iterator = {}
        for item_key in multi_train_iter:
            multi_train_flag[item_key] = True
            multi_train_iterator[item_key] = iter(multi_train_iter[item_key])

        epoch_loss = 0
        epoch_loss_count = 0
        while loop_flag(multi_train_flag.values()):
            total_loss = 0
            cross_ner_model.zero_grad()
            for item_key, item_iter in multi_train_iterator.items():
                if multi_train_flag[item_key]:
                    try:
                        train_batch = item_iter.next()

                        if item_key in ner_matrix_dir:
                            tokens_idx, chars_idx, tags_idx, tokens_mask, sentence_len, _, _ = train_batch
                            task_id = torch.LongTensor([task_vocab[item_key[0]]])
                            domain_id = torch.LongTensor([domain_vocab[item_key[1]]])
                            token_score, tags_idx, _, _ = cross_ner_model(tokens_idx, chars_idx, sentence_len, tags_idx,
                                                                          None, None, tokens_mask, task_id, domain_id,
                                                                          item_key)
                            loss = ner_criterion(token_score.view(-1, token_score.shape[-1]), tags_idx.view(-1))
                        elif item_key in lm_matrix_dir:
                            tokens_idx, chars_idx, forward_idx, backward_idx, tokens_mask, sentence_len = train_batch
                            task_id = torch.LongTensor([task_vocab[item_key[0]]])
                            domain_id = torch.LongTensor([domain_vocab[item_key[1]]])
                            final_forward, final_backward, forward_idx, backward_idx, word_len = \
                                cross_ner_model(tokens_idx, chars_idx, sentence_len, None, forward_idx, backward_idx,
                                                tokens_mask, task_id, domain_id, item_key)
                            loss = lm_criterion(final_forward.contiguous().view(-1, hyper_param.lstm_hidden),
                                                forward_idx.view(-1)) + \
                                   lm_criterion(final_backward.contiguous().view(-1, hyper_param.lstm_hidden),
                                                backward_idx.view(-1))
                            loss = loss / 2 / word_len.sum(0)

                        total_loss += loss
                        epoch_loss_count += 1
                        epoch_loss += loss.item()
                    except StopIteration:
                        multi_train_flag[item_key] = False
                        print("data {} completed down !".format(item_key))
            if total_loss:
                total_loss.backward()
                model_optimizer.step()
        print('train down loss = {0} for batch size {1}'.format(epoch_loss, epoch_loss_count))
        cross_ner_model.eval()
        cross_evaluation(multi_dev_iter, cross_ner_model, task_vocab, domain_vocab, multi_label_vocab)
        cross_evaluation(multi_test_iter, cross_ner_model, task_vocab, domain_vocab, multi_label_vocab)


def cross_evaluation(multi_iter, model, task_vocab, domain_vocab, multi_labels):
    logger = logging.getLogger("ner")
    with torch.no_grad():
        for item_key, item_iter in multi_iter.items():
            golden_label = []
            predict_label = []
            for dev_batch in item_iter:
                tokens_idx, chars_idx, tags_idx, tokens_mask, sentence_len, sentence_text, sentence_tags = dev_batch
                task_id = torch.LongTensor([task_vocab[item_key[0]]])
                domain_id = torch.LongTensor([domain_vocab[item_key[1]]])
                token_score, tags_idx, label_hat, _ = model(tokens_idx, chars_idx, sentence_len, tags_idx,
                                                            None, None, tokens_mask, task_id, domain_id, item_key)
                for p_seq, g_seq, t_seq in zip(label_hat.cpu().numpy().tolist(), tags_idx.cpu().numpy().tolist(),
                                               tokens_mask.cpu().numpy().tolist()):
                    g_labels = []
                    p_labels = []
                    for p_label_idx, g_label_idx, t_mask in zip(p_seq, g_seq, t_seq):
                        if t_mask == 1:
                            p_label = multi_labels[item_key].get_label(p_label_idx)
                            g_label = multi_labels[item_key].get_label(g_label_idx)
                            p_labels.append(p_label)
                            g_labels.append(g_label)
                    golden_label.extend(g_labels)
                    predict_label.extend(p_labels)
            precision, recall, f1 = conlleval.evaluate(golden_label, predict_label, verbose=False)
            logger.info("eval item_key {0} precision {1}, recall {2}, f1 {3}.".format(item_key, precision, recall, f1))


def loop_flag(flags):
    for flag in flags:
        if flag:
            return True
    return False


def run():
    hyper_parameter = parse_args()

    if torch.cuda.is_available():
        device = 'cuda'
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = hyper_parameter.gpu
    else:
        device = 'cpu'

    logger = logging.getLogger("ner")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if hyper_parameter.log_path:
        file_handler = logging.FileHandler(hyper_parameter.log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if hyper_parameter.mode == 'base':
        run_base_ner_model(hyper_parameter, device)
    elif hyper_parameter.mode == 'lm':
        run_lm_model(hyper_parameter, device)
    elif hyper_parameter.mode == 'cross':
        run_cross_ner_model(hyper_parameter, device)


if __name__ == '__main__':
    run()
