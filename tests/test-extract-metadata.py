import sys, os, shutil
from unittest import TestCase
from unittest.mock import patch
import unittest
import h5py, yaml
sys.path.append('../tools')
from extract_metadata import load_files, check_for_existing, process_file

class TestExtractMetrics(TestCase):

    def test_load_files_directory(self):
        tmp_dir_path = os.path.abspath('./tmp')
        os.makedirs(tmp_dir_path, exist_ok=True)
        for i in range(4):
            with open(os.path.join(tmp_dir_path, '{}.h5'.format(i)), 'w') as f:
                f.write('test')
        for i in range(3):
            with open(os.path.join(tmp_dir_path, '{}.noth5'.format(i)), 'w') as f:
                f.write('test')
        in_files, out_files = load_files(tmp_dir_path)
        self.assertEqual(len(in_files), 4)
        self.assertEqual(len(out_files), 4)
        for i in range(4):
            self.assertIn(os.path.join(tmp_dir_path, '{}.h5'.format(i)), in_files)
            self.assertIn(os.path.join(tmp_dir_path, '{}.yaml'.format(i)), out_files)
        shutil.rmtree(tmp_dir_path)

    def test_load_files_file(self):
        tmp_file_path = os.path.abspath('./tmp.h5')
        with open(tmp_file_path, 'w') as f:
            f.write('test')
        in_files, out_files = load_files(tmp_file_path)
        self.assertEqual(len(in_files), 1)
        self.assertEqual(len(out_files), 1)
        self.assertIn(tmp_file_path, in_files)
        self.assertIn(tmp_file_path.replace('h5', 'yaml'), out_files)
        os.remove(tmp_file_path)

    @patch('extract_metadata.input', return_value='none')
    def test_check_for_existing_none(self, mock_get_input):
        tmp_dir_path = os.path.abspath('./tmp')
        os.makedirs(tmp_dir_path, exist_ok=True)
        for i in range(4):
            with open(os.path.join(tmp_dir_path, '{}.h5'.format(i)), 'w') as f:
                f.write('test')
            with open(os.path.join(tmp_dir_path, '{}.yaml'.format(i)), 'w') as f:
                f.write('test')
        in_files, out_files = load_files(tmp_dir_path)
        self.assertEqual(len(in_files), 4)
        self.assertEqual(len(out_files), 4)
        in_files, out_files = check_for_existing(in_files, out_files)
        self.assertEqual(len(in_files), 0)
        self.assertEqual(len(out_files), 0)
        shutil.rmtree(tmp_dir_path)

    @patch('extract_metadata.input', return_value='all')
    def test_check_for_existing_all(self, mock_get_input):
        tmp_dir_path = os.path.abspath('./tmp')
        os.makedirs(tmp_dir_path, exist_ok=True)
        for i in range(4):
            with open(os.path.join(tmp_dir_path, '{}.h5'.format(i)), 'w') as f:
                f.write('test')
            with open(os.path.join(tmp_dir_path, '{}.yaml'.format(i)), 'w') as f:
                f.write('test')
        in_files, out_files = load_files(tmp_dir_path)
        self.assertEqual(len(in_files), 4)
        self.assertEqual(len(out_files), 4)
        in_files, out_files = check_for_existing(in_files, out_files)
        self.assertEqual(len(in_files), 4)
        self.assertEqual(len(out_files), 4)
        for i in range(4):
            self.assertIn(os.path.join(tmp_dir_path, '{}.h5'.format(i)), in_files)
            self.assertIn(os.path.join(tmp_dir_path, '{}.yaml'.format(i)), out_files)
        shutil.rmtree(tmp_dir_path)

    @patch('extract_metadata.input')
    def test_check_for_existing_some(self, mock_get_input):
        mock_get_input.side_effect = ["Y", "N", "N", "Y"]
        tmp_dir_path = os.path.abspath('./tmp')
        os.makedirs(tmp_dir_path, exist_ok=True)
        for i in range(4):
            with open(os.path.join(tmp_dir_path, '{}.h5'.format(i)), 'w') as f:
                f.write('test')
            with open(os.path.join(tmp_dir_path, '{}.yaml'.format(i)), 'w') as f:
                f.write('test')
        in_files, out_files = load_files(tmp_dir_path)
        self.assertEqual(len(in_files), 4)
        self.assertEqual(len(out_files), 4)
        in_files, out_files = check_for_existing(in_files, out_files)
        self.assertEqual(len(in_files), 2)
        self.assertEqual(len(out_files), 2)
        for i in range(4):
            if i in [0, 3]:
                self.assertIn(os.path.join(tmp_dir_path, '{}.h5'.format(i)), in_files)
                self.assertIn(os.path.join(tmp_dir_path, '{}.yaml'.format(i)), out_files)
            else:
                self.assertNotIn(os.path.join(tmp_dir_path, '{}.h5'.format(i)), in_files)
                self.assertNotIn(os.path.join(tmp_dir_path, '{}.yaml'.format(i)), out_files)

    def test_process_file_noclean(self):
        tmp_file_path = os.path.abspath('./tmp.h5')
        target_yaml_path = tmp_file_path.replace('h5', 'yaml')
        target_metadata = {
            "trial_type": "mytrialtype",
            "user_id": "luke!!",
            "date": "2019-01-01",
            "time": "00:00:00",
            "firmware_version": "1.0.0",
            "hand": "right",
            "notes": "test notes blah blah blah",
        }
        metadata = [b"mytrialtype", b"luke!!", b"2019-01-01T00-00-00", b"1.0.0", b"right", b"test notes blah blah blah"]
        with h5py.File(tmp_file_path, 'w') as f:
            f.create_dataset("metadata", data=metadata)
        process_file(tmp_file_path, target_yaml_path, clean=False)
        self.assertTrue(os.path.exists(target_yaml_path))
        with open(target_yaml_path, 'r') as f:
            result_metadata = yaml.safe_load(f)
        self.assertEqual(result_metadata, target_metadata)
        with h5py.File(tmp_file_path, 'r') as f:
            self.assertEqual(list(f["metadata"][()]), metadata)
        os.remove(tmp_file_path)
        os.remove(target_yaml_path)

    def test_process_file_clean(self):
        tmp_file_path = os.path.abspath('./tmp.h5')
        target_yaml_path = tmp_file_path.replace('h5', 'yaml')
        target_metadata = {
            "trial_type": "mytrialtype",
            "user_id": "luke!!",
            "date": "2019-01-01",
            "time": "00:00:00",
            "firmware_version": "1.0.0",
            "hand": "right",
            "notes": "test notes blah blah blah",
        }
        metadata = [b"mytrialtype", b"luke!!", b"2019-01-01T00-00-00", b"1.0.0", b"right", b"test notes blah blah blah"]
        with h5py.File(tmp_file_path, 'w') as f:
            f.create_dataset("metadata", data=metadata)
        process_file(tmp_file_path, target_yaml_path, clean=True)
        self.assertTrue(os.path.exists(target_yaml_path))
        with open(target_yaml_path, 'r') as f:
            result_metadata = yaml.safe_load(f)
        self.assertEqual(result_metadata, target_metadata)
        with h5py.File(tmp_file_path, 'r') as f:
            self.assertNotIn("metadata", f)
            self.assertNotIn("__DATATYPES__", f)
        os.remove(tmp_file_path)
        os.remove(target_yaml_path)

if __name__ == '__main__':
    unittest.main()