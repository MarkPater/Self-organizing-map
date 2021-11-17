#include "Map.hpp"

int main()
{
	std::cout << "SOM demo - start\n";

	Map som_map{ 8, 64, "optdigits.tes" };
	som_map.train();
	som_map.assign_indices();
	som_map.print_com();

	std::cout << "SOM demo - end\n";
	return 0;
}