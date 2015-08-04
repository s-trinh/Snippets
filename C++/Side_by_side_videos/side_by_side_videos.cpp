#include <iostream>

#include <opencv2/opencv.hpp>

#include <visp/vpDisplayX.h>
#include <visp/vpVideoReader.h>
#include <visp/vpVideoWriter.h>


void help() {
  std::cout << "./side_by_side_videos [options]" << std::endl <<
      "Options:" << std::endl <<
      "<path_to_video_1>                Add a path to a video." << std::endl <<
      "<path_to_video_N>                Add a path to a video." << std::endl <<
      "-f | --fps                       Specify the desired fps for the ouput file." << std::endl <<
      "-m | --mosaic <nrows:ncols>      Specify the number of rows and columns for the mosaic." << std::endl <<
      "-o | --output                    Specify the desired name for the ouput file." << std::endl <<
      "-t | --text <the text>           Specify the text to display." << std::endl <<
      "-p | --position <x:y> | <index>  Specify the location of the text to display." << std::endl <<
      "-c | --color <the color>         Specify the color of the text to display (red, blue, lightGreen, ...)." << std::endl <<
      "-c:v | --codec <fourcc code>     Specify the codec to encode the output video, unsing the FOURCC code." << std::endl <<
      "--debug                          Print debug messages." << std::endl <<
      "-h | --help                      Print this help." << std::endl;
}

int main(int argc, char **argv) {
  std::map<std::string, vpColor> mapOfColors;
  mapOfColors["black"] = vpColor::black;
  mapOfColors["white"] = vpColor::white;
  mapOfColors["lightGray"] = vpColor::lightGray;
  mapOfColors["gray"] = vpColor::gray;
  mapOfColors["darkGray"] = vpColor::darkGray;
  mapOfColors["lightRed"] = vpColor::lightRed;
  mapOfColors["red"] = vpColor::red;
  mapOfColors["darkRed"] = vpColor::darkRed;
  mapOfColors["lightGreen"] = vpColor::lightGreen;
  mapOfColors["green"] = vpColor::green;
  mapOfColors["darkGreen"] = vpColor::darkGreen;
  mapOfColors["lightBlue"] = vpColor::lightBlue;
  mapOfColors["blue"] = vpColor::blue;
  mapOfColors["darkBlue"] = vpColor::darkBlue;
  mapOfColors["yellow"] = vpColor::yellow;
  mapOfColors["cyan"] = vpColor::cyan;
  mapOfColors["orange"] = vpColor::orange;
  mapOfColors["purple"] = vpColor::purple;

  std::vector<std::string> vectorOfVideoFilenames;
  std::vector<std::string> vectorOfTexts;
  std::vector<vpImagePoint> vectorOfTextLocations;
  std::string output = "side_by_side.mpeg";
  int fps = 25;
  int nrows = 1, ncols = 2;
  vpColor colorText = vpColor::red;
  char fourcc_code[4] = {'M', 'P', 'E', 'G'};
  bool debug = false;

  if(argc < 2) {
    std::cerr << "Not enough parameters !" << std::endl;
    return -1;
  }

  for(int i = 1; i < argc;) {
    if( (std::string(argv[i]) == "-o" || std::string(argv[i]) == "--output") && i+1 < argc ) {
      output = argv[i+1];
      i+=2;
    } else if( (std::string(argv[i]) == "-m" || std::string(argv[i]) == "--mosaic") && i+1 < argc ) {
      std::string mosaic(argv[i+1]);
      std::string prefix = mosaic.substr(0, mosaic.find_first_of(":"));
      nrows = std::atoi(prefix.c_str());

      std::string suffix = mosaic.substr(mosaic.find_first_of(":")+1);
      ncols = std::atoi(suffix.c_str());
      i += 2;
    } else if( (std::string(argv[i]) == "-f" || std::string(argv[i]) == "--fps") && i+1 < argc ) {
      fps = std::atoi(argv[i+1]);
      i += 2;
    } else if( (std::string(argv[i]) == "-t" || std::string(argv[i]) == "--text") && i+1 < argc ) {
      vectorOfTexts.push_back(argv[i+1]);
      i += 2;
    } else if( (std::string(argv[i]) == "-p" || std::string(argv[i]) == "--position") && i+1 < argc ) {
      std::string position(argv[i+1]);
      if(position.find(":") != std::string::npos) {
        std::string prefix = position.substr(0, position.find_first_of(":"));
        unsigned int left = std::atoi(prefix.c_str());

        std::string suffix = position.substr(position.find_first_of(":")+1);
        unsigned int top = std::atoi(suffix.c_str());
        vectorOfTextLocations.push_back(vpImagePoint(top, left));
      } else {
        unsigned int index_position = std::atoi(position.c_str());
        vectorOfTextLocations.push_back(vpImagePoint(index_position, -1));
      }
      i += 2;
    } else if( (std::string(argv[i]) == "-c" || std::string(argv[i]) == "--color") && i+1 < argc  ) {
      std::string color = std::string(argv[i+1]);
      if(mapOfColors.find(color) != mapOfColors.end()) {
        colorText = mapOfColors[color];
      }
      i+= 2;
    } else if( (std::string(argv[i]) == "-c:v" || std::string(argv[i]) == "--codec") && i+1 < argc  ) {
      std::string str_fourcc_code = std::string(argv[i+1]);
      if(str_fourcc_code.length() == 4) {
        for(size_t cpt = 0; cpt < str_fourcc_code.length(); cpt++) {
          fourcc_code[cpt] = str_fourcc_code.at(cpt);
        }
        i+= 2;
      }
    } else if(std::string(argv[i]) == "--debug") {
      debug = true;
      i++;
    } else if(std::string(argv[i]) == "-h" || std::string(argv[i]) == "--help") {
      help();
      i++;
    } else {
      vectorOfVideoFilenames.push_back(argv[i]);
      i++;
    }
  }

  if(vectorOfVideoFilenames.empty()) {
    std::cerr << "No video file !" << std::endl;
    return -1;
  }

  if(nrows*ncols < vectorOfVideoFilenames.size()) {
    std::cerr << "Number of rows and columns is lower than the number of input video !" << std::endl;
    return -1;
  }

  if(vectorOfTexts.size() != vectorOfTextLocations.size()) {
    std::cerr << "Different numbers of text to display and text locations !" << std::endl;
    vectorOfTexts.clear();
    vectorOfTextLocations.clear();
  }

  std::vector<vpVideoReader> vectorOfVideoReaders(vectorOfVideoFilenames.size());
  std::vector<vpImage<vpRGBa> > vectorOfImages(vectorOfVideoFilenames.size());
  for(size_t i = 0; i < vectorOfVideoFilenames.size(); i++) {
    try {
      vectorOfVideoReaders[i].setFileName(vectorOfVideoFilenames[i]);
      vectorOfVideoReaders[i].open(vectorOfImages[i]);
      if(debug) {
        std::cout << "input video=" << vectorOfVideoFilenames[i] << " ; size=" <<
            vectorOfImages[i].getWidth() << "x" << vectorOfImages[i].getHeight() << std::endl;
      }
    } catch(vpException &e) {
      std::cerr << "Exception=" << e.what() << " ; input video=" << vectorOfVideoFilenames[i] << std::endl;
      return -1;
    }
  }

  unsigned int height = 0, width = 0;
  std::vector<unsigned int> vectorOfMaxWidthInCols, vectorOfMaxHeightInRows;
  //Find max height
  bool stop_for = false;
  for(int i = 0; i < nrows && !stop_for; i++) {
    unsigned int h = 0;
    for(int j = 0; j < ncols && !stop_for; j++) {
      size_t index = i*ncols + j;

      if(index >= vectorOfImages.size()) {
        stop_for = true;
      } else {
        if(vectorOfImages[index].getHeight() > h) {
          h = vectorOfImages[index].getHeight();
        }
      }
    }

    vectorOfMaxHeightInRows.push_back(h);
    height += h;
  }

  //Find max width
  for(int j = 0; j < ncols; j++) {
    unsigned int w = 0;
    for(int i = 0; i < nrows; i++) {
      size_t index = i*ncols + j;

      if(index < vectorOfImages.size() && vectorOfImages[index].getWidth() > w) {
        w = vectorOfImages[index].getWidth();
      }
    }

    vectorOfMaxWidthInCols.push_back(w);
    width += w;
  }

  std::vector<vpImagePoint> vectorOfImPts;
  //Set top left corner location
  unsigned total_current_height = 0;
  for(int i = 0; i < nrows; i++) {
    unsigned int current_height = vectorOfMaxHeightInRows[i];
    unsigned int total_current_width = 0;

    for(int j = 0; j < ncols; j++) {
      unsigned int current_width = vectorOfMaxWidthInCols[j];
      size_t index = i*ncols + j;
      unsigned int w = vectorOfImages[index].getWidth();
      unsigned int h = vectorOfImages[index].getHeight();

      vpImagePoint imPt(total_current_height + (current_height-h)/2.0, total_current_width + (current_width-w)/2.0);
      vectorOfImPts.push_back(imPt);

      if(debug) {
        std::cout << "i=" << i << " ; j=" << j << " ; top left=" << imPt << std::endl;
      }

      total_current_width += vectorOfMaxWidthInCols[j];
    }

    total_current_height += vectorOfMaxHeightInRows[i];
  }

  //Update text position
  for(std::vector<vpImagePoint>::iterator it = vectorOfTextLocations.begin(); it != vectorOfTextLocations.end(); ++it) {
    if(it->get_u() < 0) {
      size_t text_position = it->get_v();
      vpImagePoint offset(30,30);
      *it = vectorOfImPts[text_position] + offset;
    }
  }

  vpImage<vpRGBa> O;
  O.resize(height, width);

  if(debug) {
    std::cout << "output size=" << width << "x" << height << std::endl;
    std::cout << "nrows=" << nrows << " ; ncols=" << ncols << std::endl;
  }

  stop_for = false;
  for(int i = 0; i < nrows && !stop_for; i++) {
    for(int j = 0; j < ncols && !stop_for; j++) {
      size_t index = i*ncols + j;

      if(index >= vectorOfImages.size()) {
        stop_for = true;
      } else {
        O.insert(vectorOfImages[index], vectorOfImPts[index]);
      }
    }
  }

  vpDisplayX d(O, 0, 0, "Side by side");

  vpVideoWriter writer;
  writer.setFramerate(fps);
  writer.setCodec(CV_FOURCC(fourcc_code[0], fourcc_code[1], fourcc_code[2], fourcc_code[3]));
  writer.setFileName(output);
  writer.open(O);

  bool end = true;
  vpMouseButton::vpMouseButtonType button;
  try {
    do {
      end = true;
      size_t cpt = 0;

      //Clear buffer
      O = vpRGBa(0,0,0);

      //Acquire images
      for(std::vector<vpVideoReader>::iterator it = vectorOfVideoReaders.begin();
          it != vectorOfVideoReaders.end(); ++it, cpt++) {
        if(!it->end()) {
          end = false;
          it->acquire(vectorOfImages[cpt]);
        } else {
          vectorOfImages[cpt] = vpRGBa(0,0,0);
        }
      }

      //Insert images
      bool stop_for2 = false;
      for(int i = 0; i < nrows && !stop_for2; i++) {
        for(int j = 0; j < ncols && !stop_for2; j++) {
          size_t index = i*ncols + j;

          if(index >= vectorOfImages.size()) {
            stop_for2 = true;
          } else {
            O.insert(vectorOfImages[index], vectorOfImPts[index]);
          }
        }
      }

      vpDisplay::display(O);

      //Display text
      for(size_t i = 0; i < vectorOfTextLocations.size(); i++) {
        vpDisplay::displayText(O, vectorOfTextLocations[i], vectorOfTexts[i], colorText);
      }

      vpDisplay::flush(O);

      vpDisplay::getImage(O, O);
      writer.saveFrame(O);

      vpDisplay::getClick(O, button, false);
      if (button == vpMouseButton::button1) {
        end = true;
      }
    } while(!end);
  } catch(vpException &e) {
    std::cerr << "Exception=" << e.what() << std::endl;
  }

  writer.close();

  return 0;
}
