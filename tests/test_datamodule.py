import unittest

import hydra.utils
from hydra import compose, initialize
from pytorch_lightning import LightningDataModule


class TestIERelbelDataModule(unittest.TestCase):
    def setUp(self):
        # a bit like init a config file system
        with initialize(
            config_path="../configs/hydra_conf", job_name="test_app", version_base="1.2"
        ):
            cfg = compose(
                config_name="datamodule/IE/ie_rebel_small_stable.yaml",
                overrides=[
                    "+data_dir=data",
                    "+seed=42",
                    "+assets_dir=assets",
                    "+model.pretrained_model_name_or_path=gpt2",
                    "datamodule.max_num_tokens_target=2048",
                    "datamodule.linearization_class_id=fully_expanded",
                ],
            )


        self.datamodule: LightningDataModule = hydra.utils.instantiate(
            cfg, _recursive_=False
        )["datamodule"]

        self.datamodule.setup("test")

    def test_data_module_batch(self):
        test_ds = self.datamodule.data_test

        batch = next(iter(test_ds))

        self.assertEqual(batch.keys(), {"id", "text", "target", "target_ids"})

    def test_data_module_order(self):
        test_ds = self.datamodule.data_test

        batch = next(iter(test_ds))

        self.assertEqual(batch["id"], 14)


class TestIESynthieDataModule(unittest.TestCase):
    def setUp(self):
        # a bit like init a config file system
        with initialize(
            config_path="../configs/hydra_conf", job_name="test_app", version_base="1.2"
        ):
            cfg = compose(
                config_name="datamodule/IE/ie_synthie_small_stable.yaml",
                overrides=[
                    "+data_dir=data",
                    "+seed=42",
                    "+assets_dir=assets",
                    "+model.pretrained_model_name_or_path=gpt2",
                    "datamodule.max_num_tokens_target=2048",
                    "datamodule.linearization_class_id=fully_expanded",
                ],
            )

        self.datamodule: LightningDataModule = hydra.utils.instantiate(
            cfg, _recursive_=False
        )["datamodule"]

        self.datamodule.setup("test")

    def test_data_module_batch(self):
        test_ds = self.datamodule.data_test

        batch = next(iter(test_ds))

        self.assertEqual(batch.keys(), {"id", "text", "target", "target_ids"})

    def test_data_module_order(self):
        test_ds = self.datamodule.data_test

        batch = next(iter(test_ds))

        self.assertEqual(batch["id"], 856)


class TestCPPTBDataModule(unittest.TestCase):
    def setUp(self):
        # a bit like init a config file system
        with initialize(
            config_path="../configs/hydra_conf", job_name="test_app", version_base="1.2"
        ):
            cfg = compose(
                config_name="datamodule/CP/cp_ptb_stable.yaml",
                overrides=[
                    "+data_dir=data",
                    "+seed=42",
                    "+assets_dir=assets",
                    "datamodule.max_num_tokens_target=2048",
                    "datamodule.max_num_tokens_input=2048",
                    "+model.pretrained_model_name_or_path=gpt2"
                ],
            )

        self.datamodule: LightningDataModule = hydra.utils.instantiate(
            cfg, _recursive_=False
        )["datamodule"]

        self.datamodule.setup("test")

    def test_data_module_batch(self):
        test_ds = self.datamodule.data_test

        batch = next(iter(test_ds))

        self.assertEqual(batch.keys(), {"id", "text", "target", "target_ids"})

    def test_data_module_order(self):
        test_ds = self.datamodule.data_test

        batch = next(iter(test_ds))

        self.assertEqual(batch["id"], 0)


class TestEDDataModule(unittest.TestCase):

    def _load_datamodule(self, name="ace2004"):
        # a bit like init a config file system
        with initialize(
            config_path="../configs/hydra_conf", job_name="test_app", version_base="1.2"
        ):
            cfg = compose(
                config_name=f"datamodule/ED/ed_{name}_stable.yaml",
                overrides=[
                    "+data_dir=data",
                    "+seed=42",
                    "+assets_dir=assets",
                    "datamodule.max_num_tokens_target=2048",
                    "datamodule.max_num_tokens_input=2048",
                    "+model.pretrained_model_name_or_path=gpt2"
                ],
            )

        datamodule: LightningDataModule = hydra.utils.instantiate(
            cfg, _recursive_=False
        )["datamodule"]

        datamodule.setup("test")

        return datamodule


    def test_ace_2004(self):

        datamodule = self._load_datamodule("ace2004")

        test_ds = datamodule.data_test

        self.assertEqual(len(test_ds),240)

    def test_aida(self):
        datamodule = self._load_datamodule("aida")
        test_ds = datamodule.data_test
        self.assertEqual(len(test_ds),4485)

    def test_aquaint(self):
        datamodule = self._load_datamodule("aquaint")
        test_ds = datamodule.data_test
        self.assertEqual(len(test_ds),703)

    def test_clueweb(self):
        datamodule = self._load_datamodule("clueweb")
        test_ds = datamodule.data_test
        self.assertEqual(len(test_ds),11110)

    def test_msnbc(self):
        datamodule = self._load_datamodule("msnbc")
        test_ds = datamodule.data_test
        self.assertEqual(len(test_ds),651)

    def test_wiki(self):
        datamodule = self._load_datamodule("wiki")
        test_ds = datamodule.data_test
        self.assertEqual(len(test_ds),6814)

    def test_wikiLinksNED(self):
        datamodule = self._load_datamodule("wikiLinksNED")
        test_ds = datamodule.data_test
        self.assertEqual(len(test_ds),3413)


class TestDataModuleBS2(unittest.TestCase):
    def setUp(self):

        with initialize(
            config_path="../configs/hydra_conf", job_name="test_app", version_base="1.2"
        ):
            cfg = compose(
                config_name="datamodule/IE/ie_rebel_small_stable.yaml",
                overrides=[
                    "+data_dir=data",
                    "+seed=42",
                    "+assets_dir=assets",
                    "+model.pretrained_model_name_or_path=gpt2",
                    "datamodule.linearization_class_id=fully_expanded",
                    "datamodule.batch_size=2",
                ],
            )


        self.datamodule: LightningDataModule = hydra.utils.instantiate(
            cfg, _recursive_=False
        )["datamodule"]

        self.datamodule.setup("test")

    def test_data_module_batch(self):
        test_dl = self.datamodule.test_dataloader()

        self.assertEqual(test_dl.batch_size, 2)
