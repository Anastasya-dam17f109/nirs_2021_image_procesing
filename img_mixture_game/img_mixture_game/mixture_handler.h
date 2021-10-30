#pragma once
#include "mix_img_obj.h"
#include "omp.h"


struct mesh_elem {
	vector<double> lefts;
	vector<double> lefts_SKO;
	vector<double> rights;
	vector<double> rights_SKO;
	int knot_elem;
	unsigned amount = 0;
	friend std::ostream& operator << (std::ostream &re_out, const mesh_elem &elem) {
		unsigned i;
		re_out << " knot number: " << elem.knot_elem << "\n";
		re_out << "lefts " << "\n";
		for (int i = 0; i < elem.lefts.size(); ++i)
			re_out << elem.lefts[i] << " ";
		re_out << "\n";
		re_out << "rights " << "\n";
		for (i = 0; i < elem.rights.size(); ++i)
			re_out << elem.rights[i] << " ";
		return re_out;
	}
};
class mixture_handler
{
	mix_img_obj *gen_image;
	unsigned ** class_flag;
	double **my_picture;
	unsigned img_l_x;
	unsigned img_l_y;

	mesh_elem* dist_mesh;
	double * mix_shift;
	double * mix_scale;
	double * mix_weight;
	double* mix_prob;
	long double **re_comp_shifts;
	long double **re_comp_scales;

	unsigned amount_trg = 5;
	unsigned mesh_step = 10;
	unsigned window_size;
	unsigned min_trg_size = 17;
	unsigned thr_nmb = 5;
	double accuracy = 0.001;

	unsigned hyp_cl_amount = 1;
	unsigned hyp_cl_amount_mod = 1;
	bool equal_hyp_flag = true;

	double **g_i_j;
	double bic_value = 0;

	float *rfar;
	float all_mistakes = 0;
	float* mistake_mix;
	string mixture_type = "";
	std::ofstream out;
	string split_mix_filename = "D:\\splitted_image.txt";

public:
	mixture_handler(mix_img_obj* img, unsigned h_classes, double acc);
	void draw_graphics();
	void detect_result_by_mask();
	void detect_results();
	void th_detect_results(int beg, int end, int thr);
	bool get_equal_hyp_flag() {
		return equal_hyp_flag;
	}
	double get_bic_value() {
		return bic_value;
	}
	void printInformation();
	void printInformation_to_image();
	void mixture_inicalization();
	void EMalgorithm();
	void SEMalgorithm();
	void SEMalgorithm_opMP();
	void SEMalgorithm_opMP2();
	void SEMalgorithm_opMP_normal_spec();
	void SEMalgorithm_opMP_rayleigh_spec();
	void create_mesh_opMP();
	void create_mesh();
	
	double find_k_stat(double * data, int wind_size, int k_stat);
	double find_med(double* window, int wind_size);
	std::pair<int, int> partition(double* mass, int left, int right, int  ind_pivot);
	void  quickSort(double * data, int wind_size, int l, int r);

	void EMalgorithm3();
	void optimal_redraw();
	void optimal_redraw_opMP();
	void kolmogorov_optimal_redraw();
	void kolmogorov_optimal_redraw_opMP();

	void SEMoptimal_redraw();
	void SEMoptimal_redraw_1();
	void SEMoptimal_redraw_2();
	void BIC();

	~mixture_handler();
};

