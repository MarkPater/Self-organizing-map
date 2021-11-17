#include "Map.hpp"

#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <cassert>

Map::Map(int mapSize, int dimensions, const std::string& tokens_file_path,
		 bool skipFirstLine, bool labelsFront)
	: m_rows{ mapSize }
	, m_colums{ mapSize }
	, m_range_max{ m_rows + m_colums }
	, m_dimensions{ dimensions }
{
	init();
	read_data(tokens_file_path, skipFirstLine, labelsFront);
	normalize_patterns_max();
}

void Map::init()
{
	std::srand(std::time(nullptr));
	m_map.reserve(m_rows);
	for (int i{}; i < m_rows; ++i)
	{
		m_map.push_back({});
		m_map[i].reserve(m_colums);
		for (int j{}; j < m_colums; ++j)
		{
			m_map[i].push_back({});
			m_map[i][j].reserve(m_dimensions);
			for (int k{}; k < m_dimensions; ++k)
			{
				m_map[i][j].push_back(floorf(static_cast<float>(rand()) / RAND_MAX * 100) / 100);
			}
		}
	}
}

void Map::normalize_patterns_max()
{
	for (auto& column : m_data)
	{
		for (auto& token : column)
		{
			token /= 16;
		}
	}
}

void Map::train()
{
	std::cout << "train - start\n";
	for (int s{ 1 }; s <= m_steps_count; ++s)
	{
		if (s % (m_steps_count / 5) == 0)
		{
			std::cout << "Step: " << s << ";\t" << "Percentage left: " << 100 - (s * 100.0 / m_steps_count) << ".\n";
		}

		const auto pct_left{ 1 - (s * 1.0 / m_steps_count) };
		const auto curr_range{ static_cast<int>(pct_left * m_range_max) };
		const auto curr_learn_rate{ pct_left * m_learn_rate_max };
		const auto random_data_index{ std::rand() % m_data_items_count };
		const auto node_by_bmu{ closest_node(m_data[random_data_index]) };

		for (int i{}; i < m_rows; ++i)
		{
			for (int j{}; j < m_colums; ++j)
			{
				if (manhattan_distance(node_by_bmu.first, node_by_bmu.second, i, j) <= curr_range)
				{
					for (int k{}; k < m_dimensions; ++k)
					{
						m_map[i][j][k] += curr_learn_rate * (m_data[random_data_index][k] - m_map[i][j][k]);
					}
				}
			}
		}
	}
	std::cout << "train - end\n";
}

void Map::assign_indices()
{
	std::cout << "assign_indices - start\n";
	m_mapping.reserve(m_rows);
	for (int i{}; i < m_rows; ++i)
	{
		m_mapping.push_back({});
		m_mapping[i].reserve(m_colums);
		for (int j{}; j < m_colums; ++j)
		{
			m_mapping[i].push_back({});
		}
	}

	for (int i{}; i < m_data_items_count; ++i)
	{
		const auto node_by_bmu{ closest_node(m_data[i]) };
		++m_mapping[node_by_bmu.first][node_by_bmu.second][m_labels[i]];
	}
	std::cout << "assign_indices - end\n";
}

void Map::print_com() const
{
	std::cout << "Most common labels for each map node:\n";
	for (const auto& column : m_mapping)
	{
		for (const auto& matches : column)
		{
			const auto best_match = std::max_element(std::begin(matches), std::end(matches), [](const auto& left, const auto& right) {
				return left.second < right.second;
			});
			if (best_match != matches.end())
			{
				std::cout << best_match->first << " ";
			}
			else
			{
				std::cout << "- ";
			}
		}
		std::cout << "\n";
	}
}

void Map::read_data(const std::string& tokens_file_path, bool skipFirstLine, bool labelsFront)
{
	std::vector<std::string> lines;
	std::ifstream in{ tokens_file_path };
	if (in.is_open())
	{
		std::string line;
		while (std::getline(in, line))
		{
			lines.push_back(line);
		}
		in.close();
		m_data_items_count = lines.size() - (skipFirstLine ? 1 : 0);
	}
	else
	{
		assert(false, "Could not open tokens file");
		exit(1);
	}
	
	m_data.reserve(m_data_items_count);
	m_labels.reserve(m_data_items_count);
	for (int i{ skipFirstLine ? 1 : 0 }; i < lines.size(); ++i)
	{
		const auto tokens{ tokenize(lines[i])};
		m_data.push_back(tokens);
		m_labels.push_back(labelsFront ? tokens[0] : tokens[m_dimensions]);
	}
}

std::pair<int, int> Map::closest_node(const std::vector<double>& tokens) const
{
	auto min_distance{ std::numeric_limits<double>::max() };
	std::pair<int, int> result{ -1, -1 };
	for (int i{}; i < m_rows; ++i)
	{
		for (int j{}; j < m_colums; ++j)
		{
			const auto euc_distance{ euclidean_distance(m_map[i][j], tokens) };
			if (euc_distance < min_distance)
			{
				min_distance = euc_distance;
				result = std::make_pair(i, j);
			}
		}
	}
	return result;
}

std::vector<double> Map::tokenize(const std::string& tokens, const std::string& delimiter) const
{
	std::vector<double> temp;
	auto index{ 0 };
	auto fnd_digit{ tokens.find(delimiter) };
	temp.reserve(m_dimensions + 1);
	while (fnd_digit != std::string::npos)
	{
		temp.push_back(std::stod(tokens.substr(index, fnd_digit - index)));
		index = fnd_digit + delimiter.size();
		fnd_digit = tokens.find(delimiter, index);
	}
	if (index > 0)
	{
		temp.push_back(std::stoi(tokens.substr(index, tokens.size())));
	}
	return temp;
}

double Map::euclidean_distance(const std::vector<double>& first, const std::vector<double>& second) const
{
	auto result{ 0.0 };
	// -1 due to the fact that second stores labels
	if (first.size() == second.size() - 1)
	{
		for (int i{}; i < first.size(); ++i)
		{
			result += (first[i] - second[i]) * (first[i] - second[i]);
		}
	}
	else
	{
		assert(false, "Two objects have different dimensions of space");
		exit(2);
	}
	return std::sqrt(result);
}

int Map::manhattan_distance(int x1, int y1, int x2, int y2) const
{
	return std::abs(x1 - x2) + std::abs(y1 - y2);
}