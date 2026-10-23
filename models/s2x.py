import os
import json
import time
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from transformers import BertForSequenceClassification
from sklearn.metrics import f1_score

import models.gate_model as gate_model
import utils.datasets as datasets_utils
import utils.inference as inference_utils
import utils.threshold as threshold_utils
import utils.train_gate as train_gate_utils
import utils.train_pl_bert as train_pl_bert_utils
from utils.preprocessing import UNKNOWN, TOKEN_OFFSET, convert_flow_fix_size


class S2X():

    def __init__(self, args, mode="s2x-c", openworld=False):
        self.args = args
        self.mode = mode
        self.openworld = openworld
        self.device = torch.device(self.args["DEVICE"])
        self.frequency_table = None
        self.learned_table = None
        self.pl_bert = None
        self.theta_0 = None
        self.theta_1 = None
        self.theta_2 = self.args["THETA2"]
        self.now = int(datetime.now().timestamp())
        if args["IS_TRAIN"]:
            self.output_path = os.path.join(self.args["OUTPUT_PATH"], str(self.now))
            os.makedirs(self.output_path, exist_ok=True)


    def train_gate(self):
        train_df = pd.read_parquet(self.args["TRAIN_DF"])
        rows, labels = datasets_utils.get_gate_rows(train_df, self.args["DIM"])
        self.frequency_table = train_gate_utils.train_frequency_table(rows, labels, self.args["LABEL"])
        model = gate_model.GateModel(self.args["DIM"], self.args["LABEL"])
        model = train_gate_utils.train(model, self.frequency_table, rows, labels,
                                       learning_rate=self.args["GATE_LR"], epochs=self.args["GATE_EPOCH"],
                                       batch_size=self.args["GATE_BATCH_SIZE"], device=self.args["DEVICE"])
        self.learned_table = model.get_table()
        np.savez_compressed(os.path.join(self.output_path, "gate_tables.npz"),
                            learned=self.learned_table, frequency=self.frequency_table)

    def train_pl_bert(self):
        train_df = pd.read_parquet(self.args["TRAIN_DF"])
        valid_df = pd.read_parquet(self.args["VALID_DF"])
        train_data = datasets_utils.get_pl_bert_tensors(train_df, self.args["DIM"])
        valid_data = datasets_utils.get_pl_bert_tensors(valid_df, self.args["DIM"])
        self.pl_bert, seed = train_pl_bert_utils.fit(
            self.args["LABEL"], train_data, valid_data, self.device, seed=self.args["SEED"],
            epochs=self.args["PL_BERT_EPOCH"], batch_size=self.args["PL_BERT_BATCH_SIZE"],
            lr=self.args["PL_BERT_LR"], warmup_ratio=self.args["PL_BERT_WARMUP_RATIO"])
        self.pl_bert.save_pretrained(os.path.join(self.output_path, "pl_bert"))
        with open(os.path.join(self.output_path, "pl_bert", "seed.json"), "w") as f:
            json.dump({"seed": seed}, f)


    def load_model(self):
        tables = np.load(self.args["GATE_MODEL_PATH"])
        self.learned_table, self.frequency_table = tables["learned"], tables["frequency"]
        self.pl_bert = BertForSequenceClassification.from_pretrained(self.args["BERT_MODEL_PATH"])
        self.pl_bert.to(self.device).eval()

    def gate_table(self):
        return self.frequency_table if self.args["GATE_TABLE"] == "frequency" else self.learned_table

    def setting_gate_thresholds(self):
        valid_df = pd.read_parquet(self.args["VALID_DF"])
        rows, labels = datasets_utils.get_gate_rows(valid_df, self.args["DIM"])
        pred, score = inference_utils.gate_inference(self.gate_table(), rows)
        self.theta_1 = threshold_utils.get_gate_finalize_threshold(pred, score, labels, self.args["THETA1_PERCENTILE"])
        self.theta_0 = threshold_utils.get_gate_ood_threshold(score, self.args["THETA0_PERCENTILE"])
        print(f"theta0 = {self.theta_0:.6f} (open world only), theta1 = {self.theta_1:.6f}, theta2 = {self.theta_2}")

    def forward(self):
        test_df = pd.read_parquet(self.args["OPEN_TEST_DF"] if self.openworld else self.args["TEST_DF"])
        reals = test_df["y"].to_numpy(dtype=np.int64)
        gate_pred, gate_score, bert_pred, bert_score = self.scores(test_df)
        preds, deferred = self.decide(gate_pred, gate_score, bert_pred, bert_score)
        result = {"mode": self.mode, "openworld": self.openworld, "macro_f1": float(f1_score(reals, preds, average="macro")),
                  "sessions": int(len(reals))}
        if self.mode == "s2x-c":
            result["defer_rate"] = float(deferred.mean())
        if self.openworld:
            unknown, rejected = reals == UNKNOWN, preds == UNKNOWN
            result["unknown_recall"] = float(rejected[unknown].mean()) if unknown.any() else None
            result["known_false_reject"] = float(rejected[~unknown].mean()) if (~unknown).any() else None
        if self.args["MEASURE_THROUGHPUT"]:
            result["flows_per_s"] = self.measure_throughput(test_df)
        print(json.dumps(result, indent=2))
        return preds, reals

    def scores(self, df):
        gate_pred = gate_score = bert_pred = bert_score = None
        if self.mode in ("s2x-c", "s2x-g"):
            rows, _ = datasets_utils.get_gate_rows(df, self.args["DIM"])
            gate_pred, gate_score = inference_utils.gate_inference(self.gate_table(), rows)
        if self.mode in ("s2x-c", "s2x-b"):
            ids, mask, _ = datasets_utils.get_pl_bert_tensors(df, self.args["DIM"])
            proba = inference_utils.pl_bert_predict_proba(self.pl_bert, ids, mask, self.device)
            bert_pred, bert_score = proba.argmax(1), proba.max(1)
        return gate_pred, gate_score, bert_pred, bert_score

    def decide(self, gate_pred, gate_score, bert_pred, bert_score):
        if bert_pred is not None and self.openworld:
            bert_pred = np.where(bert_score < self.theta_2, UNKNOWN, bert_pred)
        if self.mode == "s2x-b":
            return bert_pred, np.ones(len(bert_pred), bool)
        reject = gate_score < self.theta_0 if self.openworld else np.zeros(len(gate_pred), bool)
        if self.mode == "s2x-g":
            return np.where(reject, UNKNOWN, gate_pred), np.zeros(len(gate_pred), bool)
        finalize = ~reject & (gate_score >= self.theta_1)
        deferred = ~reject & ~finalize
        return np.where(reject, UNKNOWN, np.where(finalize, gate_pred, bert_pred)), deferred

    def predict_one(self, prefix):
        if self.mode != "s2x-b":
            pred, score = inference_utils.gate_inference_one(self.gate_table(), inference_utils.gate_rows_one(prefix))
            if self.openworld and score < self.theta_0:
                return UNKNOWN
            if self.mode == "s2x-g" or score >= self.theta_1:
                return pred
        pred, score = inference_utils.pl_bert_inference_one(self.pl_bert, prefix, self.device)
        return UNKNOWN if self.openworld and score < self.theta_2 else pred

    def measure_throughput(self, df, flows=3000, warmup=10):
        torch.set_num_threads(1)
        prefixes = [np.asarray(convert_flow_fix_size(x, self.args["DIM"], -TOKEN_OFFSET), dtype=np.int64)
                    for x in df["x"].iloc[:flows]]
        for prefix in prefixes[:warmup]:
            self.predict_one(prefix)
        start = time.perf_counter()
        for prefix in prefixes:
            self.predict_one(prefix)
        return len(prefixes) / (time.perf_counter() - start)
