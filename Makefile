CXX      = g++
CXXFLAGS = -O2 -std=c++17 -Wall

all: bolt brandes-wiki betweeness

bolt: bolt.cpp
	$(CXX) $(CXXFLAGS) -o bolt bolt.cpp

brandes-wiki: brandes-wiki.cpp
	$(CXX) $(CXXFLAGS) -o brandes-wiki brandes-wiki.cpp

betweeness: betweeness.cpp
	$(CXX) $(CXXFLAGS) -o betweeness betweeness.cpp

# ── Quick tests ──────────────────────────────────────────────────────────────

test-wiki: bolt
	./bolt Wiki-Vote.txt 25 500 1

test-as: bolt
	./bolt as20000102.txt 25 500 1

test-astro: bolt
	./bolt ca-AstroPh.txt 25 0 0

test-synthetic: bolt
	./bolt as-22july06-synthetic.txt 25 0 0

# Run on all available graphs
run-all: bolt
	@echo "=== Wiki-Vote ==="
	./bolt Wiki-Vote.txt 25 300 1
	@echo ""
	@echo "=== AS20000102 ==="
	./bolt as20000102.txt 25 300 1
	@echo ""
	@echo "=== Email-Enron ==="
	./bolt Email-Enron.txt 25 300 1
	@echo ""
	@echo "=== oregon1_010331 ==="
	./bolt oregon1_010331.txt 25 300 1
	@echo ""
	@echo "=== CA-HepTh ==="
	./bolt CA-HepTh.txt 25 300 1
	@echo ""
	@echo "=== CA-AstroPh (no exact BC) ==="
	./bolt ca-AstroPh.txt 25 0 0
	@echo ""
	@echo "=== as-22july06-synthetic (no exact BC) ==="
	./bolt as-22july06-synthetic.txt 25 0 0

# Plot results (requires pandas + matplotlib)
plot:
	python3 plot_results.py

clean:
	rm -f bolt brandes-wiki betweeness *.o

.PHONY: all test-wiki test-as test-astro test-synthetic run-all plot clean
