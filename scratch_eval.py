import traceback
from pathlib import Path
from src.nas.fitness import FitnessEvaluator, FitnessConfig
from src.nas.search_space import SearchSpace, SearchSpaceConfig
from src.nas.encoding import GenomeEncoder
import numpy as np

def main():
    space = SearchSpace(SearchSpaceConfig())
    encoder = GenomeEncoder(space)
    gene = space.sample_population(1, np.random.default_rng(42))[0]
    candidate = encoder.gene_to_dict(gene)
    
    cfg = FitnessConfig(splits_dir=Path("data/splits"), category="bottle", device="cuda")
    evaluator = FitnessEvaluator(cfg)
    
    try:
        raw = evaluator(candidate)
        print("RAW:", raw)
        from src.nas.nsga2_engine import FITNESS_KEYS
        vals = np.array([float(raw.get(k, 0)) for i, k in enumerate(FITNESS_KEYS)])
        print("VALS:", vals)
    except Exception as e:
        traceback.print_exc()

if __name__ == "__main__":
    main()
