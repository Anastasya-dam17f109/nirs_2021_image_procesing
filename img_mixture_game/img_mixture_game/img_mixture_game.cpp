// img_mixture_game.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//

#include "pch.h"
#include <iostream>
#include "mixture_handler.h"

int main()
{
	unsigned itr_amount =5;
	unsigned i;
	double buf_bic_value = 0;
	mixture_handler** all_obj = new mixture_handler *[itr_amount-2] ;
	mixture_handler* finded_obj = nullptr;
	mix_img_obj image(110, "normal", 5, 5 , true  );
	//mix_img_obj image(110, "rayleigh", 5, 1, false );
	itr_amount = 3;
	for (i = 2; i < itr_amount; ++i) {
		all_obj[i-2] = new mixture_handler (&image, i, 0.001);
		
			if (buf_bic_value == 0) {
				buf_bic_value = all_obj[i - 2]->get_bic_value();
				finded_obj = all_obj[i - 2];
			}
			else {
				if (buf_bic_value > all_obj[i - 2]->get_bic_value()) {
					buf_bic_value = all_obj[i - 2]->get_bic_value();
					finded_obj = all_obj[i - 2];
				}
			}
		
	}
	//finded_obj = all_obj[1];
	finded_obj->printInformation();
	finded_obj->printInformation_to_image();
	finded_obj->detect_result_by_mask();
	//finded_obj->detect_results();
	finded_obj->draw_graphics();
	finded_obj = nullptr;
	for (i = 0; i < itr_amount - 2; ++i) {
		delete[] all_obj[i];
	}
	delete[] all_obj;
}

// Запуск программы: CTRL+F5 или меню "Отладка" > "Запуск без отладки"
// Отладка программы: F5 или меню "Отладка" > "Запустить отладку"

// Советы по началу работы 
//   1. В окне обозревателя решений можно добавлять файлы и управлять ими.
//   2. В окне Team Explorer можно подключиться к системе управления версиями.
//   3. В окне "Выходные данные" можно просматривать выходные данные сборки и другие сообщения.
//   4. В окне "Список ошибок" можно просматривать ошибки.
//   5. Последовательно выберите пункты меню "Проект" > "Добавить новый элемент", чтобы создать файлы кода, или "Проект" > "Добавить существующий элемент", чтобы добавить в проект существующие файлы кода.
//   6. Чтобы снова открыть этот проект позже, выберите пункты меню "Файл" > "Открыть" > "Проект" и выберите SLN-файл.
