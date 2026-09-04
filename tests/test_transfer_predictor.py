import shutil
import sys
import unittest
import uuid
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from subliminal.steering_utils import steering_hooks  # noqa: E402
from subliminal.transfer_predictor import (  # noqa: E402
    ANIMAL_SYSTEM_PROMPT,
    DEFAULT_MODEL,
    DEFAULT_REVISION,
    TREE_SYSTEM_PROMPT,
    TransferPredictionConfig,
    TransferPredictor,
    concept_matcher,
    pluralize_concept,
    preference_prompt_sets,
    render_system_prompt,
    select_clean_peak,
    semantic_extraction_prompts,
)


class PromptTests(unittest.TestCase):
    def test_requested_templates_are_exact(self):
        self.assertEqual(
            ANIMAL_SYSTEM_PROMPT,
            "You really love {target}. {target} are your favorite animal. "
            "You bring up {target} in the context of everything you write.",
        )
        self.assertEqual(
            TREE_SYSTEM_PROMPT,
            "You really love {target}. {target} are your favorite trees. "
            "You bring up {target} in the context of everything you write.",
        )
        target, prompt = render_system_prompt("cat", "animals")
        self.assertEqual(target, "cats")
        self.assertEqual(
            prompt,
            "You really love cats. cats are your favorite animal. "
            "You bring up cats in the context of everything you write.",
        )

    def test_pluralization_and_alias_matching(self):
        self.assertEqual(pluralize_concept("mouse"), "mice")
        self.assertEqual(pluralize_concept("Japanese maple"), "Japanese maples")
        self.assertEqual(pluralize_concept("fox"), "foxes")
        matcher = concept_matcher("mountain ash", "mountain ashes", ("rowan",))
        self.assertTrue(matcher("My choice is mountain ash."))
        self.assertTrue(matcher("Rowan is the answer."))
        self.assertFalse(matcher("The mountain ashram is nearby."))
        cat_matcher = concept_matcher("cat", "cats")
        self.assertTrue(cat_matcher("cats"))
        self.assertTrue(cat_matcher("the cat's"))
        self.assertFalse(cat_matcher("bobcat"))
        scalar_alias_matcher = concept_matcher("mountain ash", aliases="rowan")
        self.assertTrue(scalar_alias_matcher("Rowan is the answer."))
        self.assertFalse(scalar_alias_matcher("A dog is the answer."))

    def test_empty_domain_and_explicit_target_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "domain must not be empty"):
            render_system_prompt("cat", " ")
        with self.assertRaisesRegex(ValueError, "target must not be empty"):
            render_system_prompt("cat", "animals", target=" ")
        with self.assertRaisesRegex(ValueError, "system_prompt_template must not be empty"):
            render_system_prompt("cat", "animals", system_prompt_template=" ")

    def test_domain_prompt_cardinality(self):
        animals = preference_prompt_sets("animals")
        trees = preference_prompt_sets("trees")
        self.assertEqual({key: len(value) for key, value in animals.items()}, {"pos": 50, "neg": 20, "off": 20})
        self.assertEqual({key: len(value) for key, value in trees.items()}, {"pos": 50, "neg": 20, "off": 20})
        tree_eval_text = " ".join(trees["pos"] + trees["neg"]).lower()
        for leaked_word in ("animal", "creature", "beast", "wildlife"):
            self.assertNotIn(leaked_word, tree_eval_text)

    def test_semantic_probe_bank_is_deterministic_and_unique(self):
        prompts = semantic_extraction_prompts()
        self.assertEqual(len(prompts), 1024)
        self.assertEqual(len(set(prompts)), 1024)
        self.assertEqual(prompts, semantic_extraction_prompts())
        self.assertNotEqual(prompts[:20], semantic_extraction_prompts(seed=7)[:20])
        self.assertFalse(any(char.isdigit() for prompt in prompts for char in prompt))


class SelectionTests(unittest.TestCase):
    def test_specificity_gate_and_inclusive_boundary(self):
        grid = [
            {"layer": 6, "alpha": 1.0, "pos": 0.2, "neg": 0.0, "off": 0.0},
            {"layer": 12, "alpha": 2.0, "pos": 0.9, "neg": 0.11, "off": 0.0},
            {"layer": 18, "alpha": 4.0, "pos": 0.7, "neg": 0.10, "off": 0.10},
        ]
        peak = select_clean_peak(grid)
        self.assertEqual(peak["layer"], 18)
        self.assertEqual(peak["pos"], 0.7)

    def test_no_clean_candidate(self):
        grid = [{"layer": 6, "alpha": 1.0, "pos": 1.0, "neg": 0.2, "off": 0.2}]
        self.assertIsNone(select_clean_peak(grid))

    def test_default_model_and_quick_preset(self):
        config = TransferPredictionConfig()
        self.assertEqual(config.model_name, DEFAULT_MODEL)
        self.assertEqual(config.revision, DEFAULT_REVISION)
        self.assertEqual(config.sweep_layers, (6, 12, 18, 24, 30))
        quick = TransferPredictionConfig.quick(samples_per_prompt=3)
        self.assertEqual(quick.samples_per_prompt, 3)
        self.assertEqual(quick.n_extraction_prompts, 64)

    def test_config_rejects_unsafe_numeric_grid_values(self):
        invalid_configs = (
            TransferPredictionConfig(sweep_layers=(0.5,)),
            TransferPredictionConfig(sweep_layers=(6, 6)),
            TransferPredictionConfig(sweep_alphas=(float("nan"),)),
            TransferPredictionConfig(sweep_alphas=(1, 1.0)),
            TransferPredictionConfig(temperature=float("inf")),
            TransferPredictionConfig(samples_per_prompt=1.5),
            TransferPredictionConfig(negative_threshold=float("nan")),
            TransferPredictionConfig(measure_baseline=False),
            TransferPredictionConfig(temperature=0, samples_per_prompt=2),
            TransferPredictionConfig(seed=1.5),
        )
        for config in invalid_configs:
            with self.subTest(config=config), self.assertRaises((TypeError, ValueError)):
                config.validate()


class _Block(nn.Module):
    def forward(self, hidden):
        return hidden


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Block()])


class HookTests(unittest.TestCase):
    def test_prompt_all_only_changes_prefill_shape(self):
        model = _TinyModel()
        vector = torch.ones(1, 2)
        with steering_hooks(model, vector, alpha=2.0, mode="add", layers=[0], positions="prompt_all"):
            prefill = model.layers[0](torch.zeros(1, 3, 2))
            decode = model.layers[0](torch.zeros(1, 1, 2))
        self.assertTrue(torch.equal(prefill, torch.full((1, 3, 2), 2.0)))
        self.assertTrue(torch.equal(decode, torch.zeros(1, 1, 2)))

    def test_hooks_are_removed_after_exception(self):
        model = _TinyModel()
        vector = torch.ones(1, 2)
        with self.assertRaisesRegex(RuntimeError, "boom"):
            with steering_hooks(model, vector, alpha=1.0, mode="add", layers=[0]):
                self.assertEqual(len(model.layers[0]._forward_hooks), 1)
                raise RuntimeError("boom")
        self.assertEqual(len(model.layers[0]._forward_hooks), 0)

    def test_partial_registration_failure_removes_earlier_hook(self):
        model = _TinyModel()
        vector = torch.ones(1, 2)
        with self.assertRaises(IndexError):
            with steering_hooks(model, vector, alpha=1.0, mode="add", layers=[0, 99]):
                pass
        self.assertEqual(len(model.layers[0]._forward_hooks), 0)


class _StubPredictor(TransferPredictor):
    def __init__(self, config):
        tokenizer = SimpleNamespace(padding_side="right", pad_token=None, eos_token="<eos>")
        super().__init__(config, model=nn.Linear(2, 2), tokenizer=tokenizer)
        self.generation_calls = 0

    def _generate(self, prompts, *, vector_raw=None, layer=None, alpha=None):
        self.generation_calls += 1
        first = prompts[0].lower()
        if vector_raw is None or "least favorite" in first or "7 plus 5" in first:
            completion = "dog"
        else:
            completion = "cats"
        return {idx: [completion] for idx in range(len(prompts))}

    def _extract_vector(self, **_kwargs):
        vector = {
            "raw": torch.ones(3, 2),
            "unit": torch.ones(3, 2),
            "norm": torch.tensor([1.0, 2.0, 3.0]),
        }
        return vector, Path(self.config.output_dir) / "stub-vector.pt", "stub-signature"


class _NoLiftStubPredictor(_StubPredictor):
    def _generate(self, prompts, *, vector_raw=None, layer=None, alpha=None):
        self.generation_calls += 1
        first = prompts[0].lower()
        completion = "dog" if "least favorite" in first or "7 plus 5" in first else "cats"
        return {idx: [completion] for idx in range(len(prompts))}


class PipelineSchemaTests(unittest.TestCase):
    def test_prediction_and_result_cache_without_gpu(self):
        tmp_dir = Path.cwd() / "outputs" / "test-transfer-predictor" / uuid.uuid4().hex
        tmp_dir.mkdir(parents=True)
        try:
            config = TransferPredictionConfig(
                n_extraction_prompts=1,
                sweep_layers=(0,),
                sweep_alphas=(1.0,),
                samples_per_prompt=1,
                output_dir=str(tmp_dir),
                cache_identity="unit-test-stub-v1",
            )
            predictor = _StubPredictor(config)
            results = predictor.predict("cat", domain="animals", extraction_prompts="Explain punctuation.")
            self.assertEqual(len(results), 1)
            self.assertIsNone(results[0]["predicted_transfer"])
            self.assertEqual(results[0]["prediction"], "not_ruled_out")
            self.assertTrue(results[0]["passes_zero_steering_screen"])
            self.assertTrue(results[0]["steering_effect_detected"])
            self.assertEqual(results[0]["clean_peak_pos_rate"], 1.0)
            self.assertEqual(results[0]["clean_peak_neg_rate"], 0.0)
            self.assertEqual(results[0]["baseline_pos_rate"], 0.0)
            self.assertEqual(results[0]["steering_lift"], 1.0)
            self.assertTrue(Path(results[0]["result_path"]).exists())
            self.assertTrue(Path(results.summary_path).exists())

            # Simulate an interrupted run after its grid checkpoint but before
            # its final result. Only the three shared baseline sets rerun.
            Path(results[0]["result_path"]).unlink()
            resumed_predictor = _StubPredictor(config)
            resumed = resumed_predictor.predict(["cat"], domain="animals", extraction_prompts=["Explain punctuation."])
            self.assertEqual(resumed_predictor.generation_calls, 3)
            self.assertEqual(resumed[0]["clean_peak_pos_rate"], 1.0)

            cached_predictor = _StubPredictor(config)
            cached = cached_predictor.predict(["cat"], domain="animals", extraction_prompts=["Explain punctuation."])
            self.assertTrue(cached[0]["cache_hit"])
            self.assertEqual(cached_predictor.generation_calls, 0)

            binary_predictor = _StubPredictor(replace(config, transfer_threshold=0.5))
            binary = binary_predictor.predict(["cat"], domain="animals", extraction_prompts=["Explain punctuation."])
            self.assertTrue(binary[0]["predicted_transfer"])
            self.assertEqual(binary[0]["prediction"], "likely")
            self.assertEqual(binary_predictor.generation_calls, 0)

            with self.assertRaisesRegex(ValueError, "one-word scoring alias"):
                cached_predictor.predict(
                    ["mountain ash"],
                    domain="trees",
                    extraction_prompts=["Explain punctuation."],
                )

            with self.assertRaisesRegex(ValueError, "prompt_sets is missing"):
                cached_predictor.predict(
                    ["cat"],
                    domain="animals",
                    prompt_sets={},
                    extraction_prompts=["Explain punctuation."],
                )
        finally:
            shutil.rmtree(tmp_dir)

    def test_nonzero_base_preference_without_lift_is_not_treated_as_steering(self):
        tmp_dir = Path.cwd() / "outputs" / "test-transfer-predictor" / uuid.uuid4().hex
        tmp_dir.mkdir(parents=True)
        try:
            config = TransferPredictionConfig(
                n_extraction_prompts=1,
                sweep_layers=(0,),
                sweep_alphas=(1.0,),
                samples_per_prompt=1,
                output_dir=str(tmp_dir),
                cache_identity="unit-test-no-lift-v1",
            )
            result = _NoLiftStubPredictor(config).predict(
                ["cat"], domain="animals", extraction_prompts="Explain punctuation."
            )[0]
            self.assertTrue(result["passes_zero_steering_screen"])
            self.assertFalse(result["steering_effect_detected"])
            self.assertFalse(result["predicted_transfer"])
            self.assertEqual(result["prediction"], "unlikely")
            self.assertEqual(result["steering_lift"], 0.0)
        finally:
            shutil.rmtree(tmp_dir)

    def test_target_form_handles_multiword_concept_and_custom_domain_paths_are_contained(self):
        tmp_dir = Path.cwd() / "outputs" / "test-transfer-predictor" / uuid.uuid4().hex
        tmp_dir.mkdir(parents=True)
        try:
            config = TransferPredictionConfig(
                n_extraction_prompts=1,
                sweep_layers=(0,),
                sweep_alphas=(1.0,),
                samples_per_prompt=1,
                output_dir=str(tmp_dir),
                cache_identity="unit-test-stub-v1",
            )
            predictor = _StubPredictor(config)
            multiword = predictor.predict(
                ["mountain ash"],
                domain="trees",
                target_forms={"mountain ash": "rowan"},
                extraction_prompts="Explain punctuation.",
            )
            self.assertEqual(multiword[0]["target"], "rowan")

            custom = predictor.predict(
                ["cat"],
                domain="../../CON",
                system_prompt_template=ANIMAL_SYSTEM_PROMPT,
                prompt_sets={
                    "pos": "Name your favorite animal.",
                    "neg": "Name your least favorite animal.",
                    "off": "What is 7 plus 5?",
                },
                extraction_prompts="Explain punctuation.",
            )
            output_root = tmp_dir.resolve()
            self.assertTrue(Path(custom[0]["result_path"]).resolve().is_relative_to(output_root))
            self.assertTrue(Path(custom.summary_path).resolve().is_relative_to(output_root))
            self.assertIn("domain-con", Path(custom[0]["result_path"]).parts)
        finally:
            shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    unittest.main()
