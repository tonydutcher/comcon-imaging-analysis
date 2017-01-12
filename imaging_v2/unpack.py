import os
import argparse
import subprocess
from glob import glob

# ORGANIZATIONAL INFO
scans = {
    'comcon_localizer':
    {
        'dir': 'bold',
        'file': 'bold.nii.gz',
        'len': 250
    },
    'comcon_retrieval':
    {
        'dir': 'bold',
        'file': 'bold.nii.gz',
        'len': 343
    },
    'fieldmap':
    {
        'dir': 'fieldmap',
        'file': 'fieldmap.nii.gz',
    },
    'mprage':
    {
        'dir': 'anatomy',
        'file': 'mprage.nii.gz',
    }
}


# unpack dicoms into correct folders
def unpack_dicoms(subject_dir):
    dcm_dir = os.path.join(subject_dir,'raw','converted')

    # check existence of raw dir
    if not os.path.isdir(dcm_dir):
        raise Exception('dicom directory not found: {}'.format(dcm_dir))

    # find all sub-directories in dicom dir
    scan_dirs = [d for d in os.listdir(dcm_dir)
                if os.path.isdir(os.path.join(dcm_dir,d)) and '-' in d]

    for scan_subdir in scan_dirs:
        print "Checking {}".format(scan_subdir)
        scan_num,scan_name = scan_subdir.split('-')
        scan_name = scan_name.lower()
        scan_num = int(scan_num)
        scan_dir = os.path.join(dcm_dir,scan_subdir)

        all_files = glob(os.path.join(scan_dir,'*dcm'))
        if len(all_files) == 0: continue
        first_file = all_files[0]
        file_root = os.path.basename(first_file).split('.')[0]

        # find if scan is keep-able
        for prefix,d in scans.iteritems():
            if prefix.lower() in scan_name:

                # check scan length
                if 'len' in d:
                    print "Checking number of frames"
                    cmd = ['mri_info','--nframes', first_file]
                    print ' '.join(cmd)
                    scan_len = int(subprocess.check_output(cmd).split()[-1])
                    if d['len'] != scan_len: continue

                # convert to nifti
                print "Found: {}/{}/{}".format(d['dir'], scan_num, scan_name)
                dname = '{}_{:02d}'.format(scan_name, scan_num)
                odir  = os.path.join(subject_dir,d['dir'],dname)
                tdir  = './tmp'

                # make scan output directory
                if not os.path.isdir(odir):
                    os.makedirs(odir)

                # because dcm2nii is weird
                if os.path.isdir(tdir):
                    to_del = glob(os.path.join(tdir,'??*.*'))
                    for f in to_del: 
                        print "deleting: ", f
                        os.remove(f)
                    os.rmdir(tdir)
                os.makedirs(tdir)

                ofile = os.path.join(odir, d['file'])
                print "Converting to {}".format(ofile)
                cmd = ['dcm2nii','-4','y','-o',tdir,'-d','n','-e','n','-p','n','-f','y','-r','n','-x','n',first_file]
                print ' '.join(cmd)
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
                p.wait()

                tfile = glob(os.path.join(tdir,'*.nii.gz'))[0]
                print "temp file: {}".format(tfile)

                if prefix.lower()=='mprage':
                    # if mprage, reorient to std
                    cmd = ['fslreorient2std', tfile, ofile]
                    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
                    p.wait()
                else:
                    # move tmpfile to real location
                    os.rename(tfile,ofile)
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Organize fMRI data for analysis.')
    parser.add_argument('subject_dir', metavar="SUBJECT_DIR", type=str, 
                        help='location of subject directory')
    args = parser.parse_args()

    unpack_dicoms(args.subject_dir)
    
    