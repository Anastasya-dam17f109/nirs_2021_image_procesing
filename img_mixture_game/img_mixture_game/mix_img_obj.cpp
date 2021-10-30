#include "pch.h"
#include "mix_img_obj.h"
boost::random::mt19937 generator_{ static_cast<std::uint32_t>(time(0)) };


mix_img_obj::mix_img_obj(int img_size, string mix_t, int amount_targets, int classes, bool file_flag) {
	image_len_x = img_size * amount_targets;
	image_len_y = img_size * amount_targets;
	mixture_type = mix_t;
	amount_trg = amount_targets;
	class_amount = classes;
	if(file_flag)
		load_from_bitmap_item3();
	else
		img_generator();
	print_results();
}

//создание картинки с заданными параметрами

void mix_img_obj::img_generator() {
	boost::random::normal_distribution <> dist_norm_bcg{ 128, 37.0 };
	boost::random::normal_distribution <> dist_norm_trg{ 240, 1.5 };
	boost::random::uniform_01 <> dist_rel;

	unsigned i, j, k, l, bright_step, amount_brigh_trg, t_coord_x, t_coord_y;
	unsigned mix_number = 1;
	double sred;
	unsigned * targ_size;
	double * targ_bright;

	auto dist_gen_bcg = [&]() {
		if (mixture_type == "normal")
			return dist_norm_bcg(generator_);
		
		else {
		if (mixture_type == "rayleigh")
			return sqrt(-2 * pow(20, 2.0) *log(1 - dist_rel(generator_)));
		}
	};

	auto dist_gen_trg = [&](unsigned i) {
		if (mixture_type == "normal") {
			if (class_amount > 1)
				dist_norm_trg.param(boost::random::normal_distribution <>::param_type(targ_bright[i], 1.5*mix_number));
			return dist_norm_trg(generator_);
		}
		else {
			if (mixture_type == "rayleigh")
				return sqrt(-2 * pow(30, 2.0) *log(1 - dist_rel(generator_)));
		}
	};

	re_mix_shift = new double[class_amount + 1];
	re_mix_scale = new double[class_amount + 1];
	for (i = 0; i < class_amount + 1; ++i) {
		re_mix_shift[i] = 0.0;
		re_mix_scale[i] = 0.0;
	}

	targs = new target[amount_trg*amount_trg];
	targ_size = new unsigned[amount_trg];
	targ_bright = new double[amount_trg];
	mixture_image = new double *[image_len_x];    // массив указателей (2)

	for (int i = 0; i < image_len_x; i++) {
		mixture_image[i] = new double[image_len_x];
		for (j = 0; j < image_len_x; j++)
			mixture_image[i][j] = dist_gen_bcg();
	}

	sred = mean(mixture_image);
	bright_step = (255 - sred - 40) / class_amount;
	amount_brigh_trg = amount_trg / class_amount;
	if (amount_brigh_trg == 0)
		amount_brigh_trg = 1;

	for (i = 0; i < amount_trg; i++) {
		targ_size[i] = min_targ_size + i * 2;
		targ_bright[i] = sred + 40 + (unsigned(i / amount_brigh_trg) + 1) * bright_step;
	}
	re_mix_shift[0] = 128.0;
	re_mix_scale[0] = 37.0;
	re_mix_shift[1] = targ_bright[0];
	re_mix_scale[1] = 1.5;

	for (i = 0; i < amount_trg; i++) {
		if (mixture_type == "normal") {
			if (i > 0 && targ_bright[i] != targ_bright[i - 1]) {
				mix_number++;
				re_mix_shift[mix_number] = targ_bright[i];
				re_mix_scale[mix_number] = 1.5*mix_number;
			}
		}
		for (j = 0; j < amount_trg; j++) {
			t_coord_x = i * backg_size + backg_size / 2 - 1 - targ_size[j] / 2;
			t_coord_y = j * backg_size + backg_size / 2 - 1 - targ_size[j] / 2;
			if (mixture_type == "normal")
				targs[i*amount_trg + j].brightness = targ_bright[i];
			else
				targs[i*amount_trg + j].brightness = 30;
			targs[i*amount_trg + j].size = targ_size[j];
			targs[i*amount_trg + j].x = t_coord_x;
			targs[i*amount_trg + j].y = t_coord_y;

			for (k = 0; k < targ_size[j]; k++) {
				for (l = 0; l < targ_size[j]; l++) {
					mixture_image[t_coord_x + k][t_coord_y + l] = dist_gen_trg(i);
					//cout << mixture_image[t_coord_x + k][t_coord_y + l] << endl;
				}
			}
		}
	}

	for (i = 0; i < amount_trg*amount_trg; i++) {
		for (j = 1; j < class_amount + 1; j++)
			if (targs[i].brightness == re_mix_shift[j])
				targs[i].mix_type = j + 1;
	}
	if (mixture_type == "rayleigh") {
		re_mix_shift[0] = 20.0;
		re_mix_scale[0] = 20.0;
		re_mix_shift[1] = 30;
		re_mix_scale[1] = 30;
	}
	delete[] targ_size;
	delete[] targ_bright;
}

//каритнка_поля_папка_image_data

void mix_img_obj::load_from_bitmap_item1() {
	CImage image; 
	CImage image_buf;
	image_buf.Load(_T("D:\\image_data\\__FR1.JPG_007.DBF.BMP"));
	item_type = L"item1";
	int buf_amount = 0;
	int y_len, x_len, i,j,k;
	mixture_type = "lognormal";
	class_amount = 4;

	re_mix_shift = new double[class_amount + 1];
	re_mix_scale = new double[class_amount + 1];
	mask_list = new LPCTSTR[class_amount + 1];
	for (i = 0; i < class_amount + 1; ++i) {
		re_mix_shift[i] = 0.0;
		re_mix_scale[i] = 0.0;
		mask_list[i] = L"";
	}
	LPCTSTR * cstr = new LPCTSTR[6];
	cstr[0] = L"D:\\image_data\\__FR1.jpg_000.DBF.BMP";
	cstr[1] = L"D:\\image_data\\__FR1.jpg_001.DBF.BMP";
	cstr[2] = L"D:\\image_data\\__FR1.jpg_002.DBF.BMP";
	cstr[3] = L"D:\\image_data\\__FR1.jpg_003.DBF.BMP";
	cstr[4] = L"D:\\image_data\\__FR1.JPG_0010.DBF.BMP";
	/*cstr[4] = L"D:\\image_data\\__FR1.jpg_004.DBF.BMP";
	cstr[5] = L"D:\\image_data\\__FR1.jpg_005.DBF.BMP";*/
	double n_length = 0;
	for (k = 0; k < class_amount+1; ++k) {
		image.Load(cstr[k]);
		y_len = image.GetHeight();
		x_len = image.GetWidth();

		for ( i = 0; i < y_len; i++) 
			for (j = 0; j < x_len; j++) {
				if (k == 4) {
					re_mix_shift[k] += double(GetGValue(image.GetPixel(j, i)))*(1-int(GetGValue(image_buf.GetPixel(j, i)))/255);
					//cout << int(GetGValue(image_buf.GetPixel(j, i))) / 255 << endl;
					buf_amount += (1-int(GetGValue(image_buf.GetPixel(j, i))) / 255);
				}
				else
				re_mix_shift[k] += double(GetGValue(image.GetPixel(j, i)));
			}
		if(k==4)
			re_mix_shift[k] /= double(buf_amount);
		else
		re_mix_shift[k] /= double(y_len * x_len);
		cout << "buf_amount " << buf_amount <<" " << y_len * x_len <<endl;
		for (i = 0; i < y_len; i++)
			for (j = 0; j < x_len; j++) {
				if (k == 4) {
					re_mix_scale[k] += pow((double((GetGValue(image.GetPixel(j, i)))) - re_mix_shift[k])*(1-int(GetGValue(image_buf.GetPixel(j, i))) / 255), 2);
				}
				else
				re_mix_scale[k] += pow(double(GetGValue(image.GetPixel(j, i))) - re_mix_shift[k], 2);
				
			}
		if (k == 4)
		re_mix_scale[k] = re_mix_scale[k] / double(buf_amount);
		else
			re_mix_scale[k] = re_mix_scale[k] / (double(y_len * x_len));
		
		re_mix_scale[k] = sqrt(log(re_mix_scale[k] / (re_mix_shift[k] * re_mix_shift[k]) + 1.0));
		re_mix_shift[k] = log(re_mix_shift[k] / exp(re_mix_scale[k] * re_mix_scale[k] / 2.0));
		image.Detach();
	}
	
	image.Load(_T("D:\\image_data\\__FR1_modified.jpg"));
	y_len = image.GetHeight();
	x_len = image.GetWidth();
	image_len_x = (y_len / 10) * 10;

	image_len_y = (x_len / 10) * 10;
	mixture_image = new double *[image_len_x];    // массив указателей (2)
	for (int i = 0; i < image_len_x; i++) {
		mixture_image[i] = new double[image_len_y];     // инициализация указателей
	}
	for (int i = 0; i < image_len_x; i++) {
		for (int j = 0; j < image_len_y; j++) {
			mixture_image[image_len_x - i - 1][j] = double(GetGValue(image.GetPixel(j, i)));
		}// инициализация указателей
	}
	mask_list[0] = L"D:\\image_data\\new_mask_class_10.BMP";
	mask_list[1] = L"D:\\image_data\\new_mask.BMP";
	mask_list[2] = L"D:\\image_data\\new_mask_class_3.BMP";
	out.open(filename_gen_image);
	for (int i = 0; i < image_len_x; i++) {
		for (int j = 0; j < image_len_x; j++)
			out << mixture_image[i][j] << " ";
		out << std::endl;
	}
	out.close();
	/*cout << "pixel " << float(GetGValue( image.GetPixel(0, 0)))<<" "<< GetRValue(image.GetPixel(0, 0)) << " " << GetBValue(image.GetPixel(0, 0)) << endl;
	printf("%i", GetGValue(image.GetPixel(0, 0)));*/
}

///каритнка_поля_папка_Sar_China

void mix_img_obj::load_from_bitmap_item2() {
	CImage image;
	CImage image_mask;
	item_type = "item2";
	image_mask.Load(_T("D:\\_SAR_China\\__CHENDGU_FR1.TIF.JPG_009.DBF.BMP"));
	int buf_amount = 0;
	int y_len, x_len, i, j, k;
	mixture_type = "lognormal";
	class_amount = 4;

	re_mix_shift = new double[class_amount + 1];
	re_mix_scale = new double[class_amount + 1];
	for (i = 0; i < class_amount + 1; ++i) {
		re_mix_shift[i] = 0.0;
		re_mix_scale[i] = 0.0;
	}
	LPCTSTR * cstr = new LPCTSTR[6];
	cstr[0] = L"D:\\_SAR_China\\__Chendgu_fr1.tif.JPG_000.DBF.BMP";
	cstr[1] = L"D:\\_SAR_China\\__Chendgu_fr1.tif.JPG_004.DBF.BMP";
	cstr[2] = L"D:\\_SAR_China\\__Chendgu_fr1.tif.JPG_007.DBF.BMP";
	cstr[3] = L"D:\\_SAR_China\\__Chendgu_fr1.tif.JPG_008.DBF.BMP";

	cstr[4] = L"D:\\_SAR_China\\__Chendgu_fr1.tif.JPG_003.DBF.BMP";
	/*cstr[4] = L"D:\\image_data\\__FR1.jpg_004.DBF.BMP";
	cstr[5] = L"D:\\image_data\\__FR1.jpg_005.DBF.BMP";*/
	double n_length = 0;
	for (k = 0; k < class_amount + 1; ++k) {
		image.Load(cstr[k]);
		y_len = image.GetHeight();
		x_len = image.GetWidth();

		for (i = 0; i < y_len; i++)
			for (j = 0; j < x_len; j++) {
				if (k == 4) {
					re_mix_shift[k] += double(GetGValue(image.GetPixel(j, i)))*(1 - int(GetGValue(image_mask.GetPixel(j, i))) / 255);
					//cout << int(GetGValue(image_buf.GetPixel(j, i))) / 255 << endl;
					buf_amount += (1 - int(GetGValue(image_mask.GetPixel(j, i))) / 255);
				}
				else
					re_mix_shift[k] += double(GetGValue(image.GetPixel(j, i)));
			}
		if (k == 4)
			re_mix_shift[k] /= double(buf_amount);
		else
			re_mix_shift[k] /= double(y_len * x_len);
		cout << "buf_amount " << buf_amount << " " << y_len * x_len << endl;
		for (i = 0; i < y_len; i++)
			for (j = 0; j < x_len; j++) {
				if (k == 4) {
					re_mix_scale[k] += pow((double((GetGValue(image.GetPixel(j, i)))) - re_mix_shift[k])*(1 - int(GetGValue(image_mask.GetPixel(j, i))) / 255), 2);
				}
				else
					re_mix_scale[k] += pow(double(GetGValue(image.GetPixel(j, i))) - re_mix_shift[k], 2);

			}
		if (k == 4)
			re_mix_scale[k] = re_mix_scale[k] / double(buf_amount);
		else
			re_mix_scale[k] = re_mix_scale[k] / (double(y_len * x_len));

		re_mix_scale[k] = sqrt(log(re_mix_scale[k] / (re_mix_shift[k] * re_mix_shift[k]) + 1.0));
		re_mix_shift[k] = log(re_mix_shift[k] / exp(re_mix_scale[k] * re_mix_scale[k] / 2.0));
		image.Detach();
	}

	//image.Load(_T("D:\\_SAR_China\\__Chendgu_fr1.tif.JPG"));
	image.Load(_T("D:\\_SAR_China\\__Chendgu_fr1_sec_part.tif.jpg"));
	
	y_len = image.GetHeight();
	x_len = image.GetWidth();
	if (y_len > x_len)
		image_len_x = x_len;
	else
		image_len_x = y_len;
	image_len_x = (image_len_x / 10) * 10;
	mixture_image = new double *[image_len_x];    // массив указателей (2)
	for (int i = 0; i < image_len_x; i++) {
		mixture_image[i] = new double[image_len_x];     // инициализация указателей
	}
	for (int i = 0; i < image_len_x; i++) {
		for (int j = 0; j < image_len_x; j++) {
			mixture_image[image_len_x - i - 1][j] = double(GetGValue(image.GetPixel(j, i)));
		}// инициализация указателей
	}

	out.open(filename_gen_image);
	for (int i = 0; i < image_len_x; i++) {
		for (int j = 0; j < image_len_x; j++)
			out << mixture_image[i][j] << " ";
		out << std::endl;
	}
	out.close();
	/*cout << "pixel " << float(GetGValue( image.GetPixel(0, 0)))<<" "<< GetRValue(image.GetPixel(0, 0)) << " " << GetBValue(image.GetPixel(0, 0)) << endl;
	printf("%i", GetGValue(image.GetPixel(0, 0)));*/
}

// каритнка_поля_папка_Sar_Kubinka

void mix_img_obj::load_from_bitmap_item3() {
	CImage image;
	CImage image_mask;
	item_type = L"item3";
	image_mask.Load(_T("D:\\_SAR_Kubinka\\__FRAGM1856X.JPG_017.DBF.BMP"));
	int buf_amount = 0;
	int y_len, x_len, i, j, k;
	mixture_type = "lognormal";
	class_amount = 4;
	
	re_mix_shift = new double[class_amount + 1];
	re_mix_scale = new double[class_amount + 1];
	mask_list = new LPCTSTR[class_amount + 1];
	for (i = 0; i < class_amount + 1; ++i) {
		re_mix_shift[i] = 0.0;
		re_mix_scale[i] = 0.0;
		mask_list[i] = L"";
	}
	LPCTSTR * cstr = new LPCTSTR[class_amount + 1];
	cstr[0] = L"D:\\_SAR_Kubinka\\__FRAGM1856X.JPG_000.DBF.BMP";// темные участки
	cstr[1] = L"D:\\_SAR_Kubinka\\__FRAGM1856X.JPG_005.DBF.BMP";// земля голая
	//cstr[2] = L"D:\\_SAR_Kubinka\\__FRAGM1856X.JPG_008.DBF.BMP";
	cstr[2] = L"D:\\_SAR_Kubinka\\__FRAGM1856X.JPG_006.DBF.BMP"; // лес яркий

	//cstr[4] = L"D:\\_SAR_Kubinka\\__FRAGM1856X.JPG_016.DBF.BMP";
	//cstr[3] = L"D:\\_SAR_Kubinka\\__FRAGM1856X.JPG_011.DBF.BMP"; //  поле
	cstr[3] = L"D:\\_SAR_Kubinka\\__FRAGM1856X.JPG_019.DBF.BMP";
	cstr[4] = L"D:\\_SAR_Kubinka\\__FRAGM1856X.JPG_020.DBF.BMP"; // яркие участки
	double n_length = 0;
	for (k = 0; k < class_amount + 1; ++k) {
		image.Load(cstr[k]);
		y_len = image.GetHeight();
		x_len = image.GetWidth();

		for (i = 0; i < y_len; i++)
			for (j = 0; j < x_len; j++) {
				//if (k == 4) {
					//re_mix_shift[k] += double(GetGValue(image.GetPixel(j, i)))*(1 - int(GetGValue(image_mask.GetPixel(j, i))) / 255);
					//cout << int(GetGValue(image_buf.GetPixel(j, i))) / 255 << endl;
					//buf_amount += (1 - int(GetGValue(image_mask.GetPixel(j, i))) / 255);
				/*}
				else*/
					re_mix_shift[k] += double(GetGValue(image.GetPixel(j, i)));
			}
		//if (k == 4){}
			//re_mix_shift[k] /= double(buf_amount);
		//else
			re_mix_shift[k] /= double(y_len * x_len);
		//cout << "buf_amount " << buf_amount << " " << y_len * x_len << endl;
		for (i = 0; i < y_len; i++)
			for (j = 0; j < x_len; j++) {
				//if (k == 4) {
					//re_mix_scale[k] += pow((double((GetGValue(image.GetPixel(j, i)))) - re_mix_shift[k])*(1 - int(GetGValue(image_mask.GetPixel(j, i))) / 255), 2);
				//}
				//else
					re_mix_scale[k] += pow(double(GetGValue(image.GetPixel(j, i))) - re_mix_shift[k], 2);

			}
		//if (k == 4){}
			//re_mix_scale[k] = re_mix_scale[k] / double(buf_amount);
		//else
			re_mix_scale[k] = re_mix_scale[k] / (double(y_len * x_len));

		re_mix_scale[k] = sqrt(log(re_mix_scale[k] / (re_mix_shift[k] * re_mix_shift[k]) + 1.0));
		re_mix_shift[k] = log(re_mix_shift[k] / exp(re_mix_scale[k] * re_mix_scale[k] / 2.0));
		image.Detach();
	}

	//image.Load(_T("D:\\_SAR_China\\__Chendgu_fr1.tif.JPG"));
	image.Load(_T("D:\\_SAR_Kubinka\\__FRAGM1856X.JPG"));

	y_len = image.GetHeight();
	x_len = image.GetWidth();
	image_len_x = (y_len / 10) * 10;
	
	image_len_y = (x_len / 10) * 10;
	mixture_image = new double *[image_len_x];    // массив указателей (2)
	for (int i = 0; i < image_len_x; i++) {
		mixture_image[i] = new double[image_len_y];     // инициализация указателей
	}
	for (int i = 0; i < image_len_x; i++) {
		for (int j = 0; j < image_len_y; j++) {
			mixture_image[image_len_x - i - 1][j] = double(GetGValue(image.GetPixel(j, i)));
		}// инициализация указателей
	}
	mask_list[1] = L"D:\\_SAR_Kubinka\\new_mask_class_1.BMP";
	mask_list[2] = L"D:\\_SAR_Kubinka\\new_mask_class_2.BMP";
	mask_list[3] = L"D:\\_SAR_Kubinka\\new_mask_class_3.BMP";
	out.open(filename_gen_image);
	for (int i = 0; i < image_len_x; i++) {
		for (int j = 0; j < image_len_y; j++)
			out << mixture_image[i][j] << " ";
		out << std::endl;
	}
	out.close();
	
}

//вывод сведений об изображении

void  mix_img_obj::print_results() {
	unsigned i, j;
	out.open(filename_gen_image);
	for (i = 0; i < image_len_x; i++) {
		for (j = 0; j < image_len_y; j++)
			out << mixture_image[i][j] << " ";
		out << std::endl;
	}
	out.close();
	cout << " generated image params:" << "\n";
	cout << "mixture type: " << mixture_type << "\n";
	cout << "size: " << image_len_x << " " << image_len_y << "\n";
	cout << " mix components amount: " << class_amount + 1 << "\n";
	cout << "re_mix_shift values:" << endl;
	for (i = 0; i < class_amount + 1; i++)
		cout << re_mix_shift[i] << "  ";
	cout << "\n";

	cout << "re_mix_scale values:" << "\n";
	for (i = 0; i < class_amount + 1; i++)
		cout << re_mix_scale[i] << "  ";
	cout << endl;
}

//вычисление среднего арифметического

int mix_img_obj::mean(double** data) {
	double result = 0;
	for (int k = 0; k < image_len_x; k++) {
		for (int l = 0; l < image_len_y; l++)
			result += data[k][l];
	}
	return int(result / (image_len_x*image_len_y));
}

//
string  mix_img_obj::get_filename() {
	return filename_gen_image;
}

//

double** mix_img_obj::get_image() {
	return mixture_image;
}

//

std::pair<int, int> mix_img_obj::get_image_len() {
	return std::pair<int, int>(image_len_x, image_len_y);
}

//

target*  mix_img_obj::get_targets() {
	return targs;
}

//

unsigned mix_img_obj::get_min_targ_size() {
	return min_targ_size
		;
}


//деструктор

mix_img_obj::~mix_img_obj() {
	unsigned i;
	cout << "dell" << endl;
	for (int i = 0; i < image_len_x*image_len_x; i++) {
		if (i < image_len_x)
			delete[] mixture_image[i];
	}
	delete[] mixture_image;
	delete[] re_mix_shift;
	delete[] re_mix_scale;
	delete[] targs;
}
