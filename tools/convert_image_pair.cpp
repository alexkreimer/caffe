// This program converts a set of images to a lmdb/leveldb by storing them
// as Datum proto buffers.
// Usage:
//   convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.JPEG 7
//   ....

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

DEFINE_bool(gray, false,
    "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false,
    "Randomly shuffle the order of images and their labels");
DEFINE_string(backend, "lmdb",
        "The backend {lmdb, leveldb} for storing the result");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");
DEFINE_bool(check_size, false,
    "When this option is on, check that all the datum have the same size");
DEFINE_bool(encoded, false,
    "When this option is on, the encoded image will be save in datum");
DEFINE_string(encode_type, "",
    "Optional: What type should we encode the image as ('png','jpg',...).");

std::vector<string>
split(const char *str, char c = ' ')
{
    std::vector<string> result;
    do
    {
        const char *begin = str;

        while(*str != c && *str)
            str++;
        result.push_back(string(begin, str));
    } while (0 != *str++);

    return result;
}

int main(int argc, char** argv) {
#ifdef USE_OPENCV
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a set of images to the leveldb/lmdb\n"
        "format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_imageset [FLAGS] LISTFILE DB_NAME_IMAGES DB_NAME_LABELS\n"
        "The ImageNet dataset for the training demo is at\n"
        "    http://www.image-net.org/download-images\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 3) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_imageset");
    return 1;
  }

  const bool is_color = !FLAGS_gray;
  const bool check_size = FLAGS_check_size;
  const bool encoded = FLAGS_encoded;
  const string encode_type = FLAGS_encode_type;

  std::ifstream infile(argv[1]);
  typedef std::tuple<std::string, std::string, double> value_type;
  std::vector<value_type> lines;
  std::string line;

  int line_no = 0;
  while (std::getline(infile, line)) {
      auto tokens = split(line.c_str());
      if (tokens.size() != 3)
      {
          std::stringstream message;
          message << "line " << line_no << ": syntax error, can't find 3 tokens";
          throw std::runtime_error(message.str());
      }
      lines.push_back(std::make_tuple(tokens[0], tokens[1], std::stod(tokens[2])));
  }
  if (FLAGS_shuffle) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    shuffle(lines.begin(), lines.end());
  }
  LOG(INFO) << "A total of " << lines.size() << " images.";

  if (encode_type.size() && !encoded)
    LOG(INFO) << "encode_type specified, assuming encoded=true.";

  int resize_height = std::max<int>(0, FLAGS_resize_height);
  int resize_width = std::max<int>(0, FLAGS_resize_width);

  Datum datum_images;
  int count = 0;
  int data_size = 0;
  bool data_size_initialized = false;

  std::cout << "writing labels db" << std::endl;
  scoped_ptr<db::DB> db1(db::GetDB(FLAGS_backend));
  db1->Open(argv[3], db::NEW);
  scoped_ptr<db::Transaction> txn1(db1->NewTransaction());
  
  for (int line_id = 0; line_id < lines.size(); ++line_id) {
    // the set_label() is used only for sanity checks
    Datum datum_label;
    datum_label.clear_data();
    datum_label.clear_float_data();
    datum_label.set_encoded(false);
    datum_label.set_channels(1);
    datum_label.set_height(1);
    datum_label.set_width(1);
    datum_label.add_float_data(0);
    auto label = (float)std::get<2>(lines[line_id]);    
    datum_label.set_float_data(0, label);
    datum_label.set_label(line_id);
    
    // sequential
    string key_str = caffe::format_int(line_id, 8) + "_" + std::get<0>(lines[line_id]) + "_" + std::get<1>(lines[line_id]);
    datum_label.set_param(key_str);

    string out1;
    CHECK(datum_label.SerializeToString(&out1));
    txn1->Put(key_str, out1);
    
    if (++count % 1000 == 0) {
      txn1->Commit();
      txn1.reset(db1->NewTransaction());
      LOG(INFO) << "Processed " << count << " files.";
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
    txn1->Commit();
    LOG(INFO) << "Processed " << count << " files.";
  }
  
  // Create new DB
  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  db->Open(argv[2], db::NEW);
  scoped_ptr<db::Transaction> txn(db->NewTransaction());
  

  // prepare the label blob
  std::cout << "writing images lmdb..." << std::endl;
  count = 0;
  for (int line_id = 0; line_id < lines.size(); ++line_id) {
    // sequential
    string key_str = caffe::format_int(line_id, 8) + "_" + std::get<0>(lines[line_id]) + "_" + std::get<1>(lines[line_id]);
      
    // the set_label() is used only for sanity checks
    bool status;
    std::string enc = encode_type;
    auto image_files = std::make_pair(std::get<0>(lines[line_id]), std::get<1>(lines[line_id]));
    status = ReadImagesToDatum(image_files, line_id, resize_height, resize_width, is_color, &datum_images);
    if (status == false)
    {
        LOG(ERROR) << "Skeep image pair " << key_str << std::endl;
        continue;
    }
    if (check_size) {
      if (!data_size_initialized) {
        data_size = datum_images.channels() * datum_images.height() * datum_images.width();
        data_size_initialized = true;
      } else {
        const std::string& data = datum_images.data();
        CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
            << data.size();
      }
    }
    datum_images.set_label(line_id);
    datum_images.set_param(key_str);
    // Put in db
    string out;
    CHECK(datum_images.SerializeToString(&out));
    txn->Put(key_str, out);

    if (++count % 1000 == 0) {
      // Commit db
      txn->Commit();
      txn.reset(db->NewTransaction());
      LOG(INFO) << "Processed " << count << " files.";
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
    txn->Commit();
    LOG(INFO) << "Processed " << count << " files.";
  }
  
#else
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  return 0;
}
