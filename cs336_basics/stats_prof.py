import pstats
pstats.f8 = lambda x: f"{x:8.6f}"
p = pstats.Stats('bpe.prof')
p.strip_dirs().sort_stats('cumulative').print_stats("bpe")
