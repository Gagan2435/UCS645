CXX = g++
CXXFLAGS = -std=c++17 -Wall -O2 -fopenmp -march=native
TARGET = correlate
SOURCES = main.cpp correlate.cpp
OBJECTS = $(SOURCES:.cpp=.o)
HEADERS = correlate.h

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS)

%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET) results.csv

.PHONY: all clean
