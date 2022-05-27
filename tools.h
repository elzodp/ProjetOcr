
struct Image {
	// On doit rester en float vu qu'on travaille avec des floats
	int width;
	int heigth;
	int type;
	size_t size=0;
	uint8_t* data=NULL;

	Image(const char* filename);
	Image(int width, int heigth, int type);
	Image(const Image& Im);
	~Image();



};

void Image () {

	std::string image_path = "something";


}
