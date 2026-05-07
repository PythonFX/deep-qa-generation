from pathlib import Path

from rag_qa_builder.config import AppConfig
from rag_qa_builder.pipeline import Pipeline


def test_pipeline_generates_final_dataset(tmp_path: Path) -> None:
    config = AppConfig()
    config.llm.enabled = False
    config.qa_generation.target_size = 20
    pipeline = Pipeline(
        input_path=Path(__file__).parent / "fixtures",
        output_dir=tmp_path,
        config=config,
        dry_run=True,
    )
    result = pipeline.generate_all()
    assert result["documents"]
    assert result["concepts"]
    assert result["facts"]
    assert (tmp_path / "dataset.final.jsonl").exists()

