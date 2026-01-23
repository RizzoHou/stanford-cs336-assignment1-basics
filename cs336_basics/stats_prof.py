import pstats
pstats.f8 = lambda x: f"{x:8.6f}"
p = pstats.Stats('bpe-100mb-v2.prof')
p.strip_dirs().sort_stats('tot').print_stats("bpe")
