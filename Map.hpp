#ifndef MAP_H
#define MAP_H

#include <iostream>
#include <string>
#include <vector>
#include <map>

class Map
{
public:
	explicit Map(int mapSize, int dimensions, const std::string& tokens_file_path,
				 bool skipFirstLine = false, bool lablesFront = false);
	~Map() = default;

	void train();
	void assign_indices();
	void print_com() const;

private:
	void init();
	void normalize_patterns_max();
	void read_data(const std::string& tokens_file_path, bool skipFirstLine, bool labelsFront);
	std::pair<int, int> closest_node(const std::vector<double>& tokens) const;
	std::vector<double> tokenize(const std::string& tokens, const std::string& delimiter = ",") const;
	double euclidean_distance(const std::vector<double>& first, const std::vector<double>& second) const;
	int manhattan_distance(int x1, int y1, int x2, int y2) const;

	std::vector<int> m_labels;
	std::vector<std::vector<double>> m_data;
	std::vector<std::vector<std::vector<double>>> m_map;
	std::vector<std::vector<std::map<int, int>>> m_mapping;

	int m_rows{};
	int m_colums{};
	int m_range_max{};
	int m_data_items_count{};
	const int m_steps_count{ 100000 };
	const double m_learn_rate_max{ 0.5 };
	const int m_dimensions{ 64 };
};

#endif // MAP_H