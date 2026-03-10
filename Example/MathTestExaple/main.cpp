#include "math/matrix.hpp"
#include "math/Vector.hpp"
#include <iostream>

/**
*-8.42231 -2.64442 -1.49155
3.38562 1.98684 7.69219
3.77723 7.33149 0.81639
*/

int main()
{
	mat3<double> dDs_dx[12];
	mat3<double> dF_dx[12];
	mat3<double> Dm_inv(-8.42231, -2.64442, -1.49155,
						3.38562, 1.98684, 7.69219,
						3.77723, 7.33149, 0.81639);
	mat3<double> Dm_invT = mat3<double>::transpose(Dm_inv);

	for (int i = 0; i < 12; i++)
	{
		dDs_dx[i] = mat3<double>(double(0));
		dF_dx[i] = mat3<double>(double(0));
	}

	for (int i = 1; i < 4; i++)
	{
		for (int j = 0; j < 3; j++)
			dDs_dx[i * 3 + j](j, (i - 1)) = 1;
	}

	for (int i = 0; i < 3; i++)
		dDs_dx[i] = static_cast<double>(-1) * (dDs_dx[i + 3] + dDs_dx[i + 6] + dDs_dx[i + 9]);

	for (int i = 0; i < 12; i++)
	{
		dF_dx[i] = dDs_dx[i] * Dm_inv;
	}

	for (int i = 0; i < 12; i++)
	{
		mat3<double> tmp_mat = dDs_dx[i];
		printf("dDs_dx%i \n", i);
		for (int j = 0; j < 3; j++)
		{
			for (int k = 0; k < 3; k++)
				std::cout << tmp_mat(j, k) << ' ';
			std::cout << std::endl;
		}
	}

	for (int i = 0; i < 12; i++)
	{
		mat3<double> tmp_mat = dF_dx[i];
		printf("dF_dx%i \n", i);
		for (int j = 0; j < 3; j++)
		{
			for (int k = 0; k < 3; k++)
				std::cout << tmp_mat(j, k) << ' ';
			std::cout << std::endl;
		}
	}

	std::cout << std::endl << "method2: " << std::endl;
	Vec3d g[4];
	g[1] = Dm_invT.column(0);
	g[2] = Dm_invT.column(1);
	g[3] = Dm_invT.column(2);
	g[0] = -(g[1] + g[2] + g[3]);

	#pragma unroll
	for (int a = 0; a < 4; ++a) {
	#pragma unroll
		for (int c = 0; c < 3; ++c) {
			mat3<double> dF(double(0));
			for (int j = 0; j < 3; ++j) {
				dF(c, j) = g[a][j];
			}
			dF_dx[a * 3 + c] = dF;
		}
	}
	for (int i = 0; i < 12; i++)
	{
		mat3<double> tmp_mat = dF_dx[i];
		printf("dF_dx%i \n", i);
		for (int j = 0; j < 3; j++)
		{
			for (int k = 0; k < 3; k++)
				std::cout << tmp_mat(j, k) << ' ';
			std::cout << std::endl;
		}
	}
	return 0;
}