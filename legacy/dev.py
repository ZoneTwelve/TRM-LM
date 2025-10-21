from models.trm import TinyRecursiveReasoningModel_ACTV1, TinyRecursiveReasoningModel_ACTV1Config


conf = TinyRecursiveReasoningModel_ACTV1Config(
    batch_size=32,
    seq_len=10,
    puzzle_emb_ndim=0
    num_puzzle_identifiers=0
)
