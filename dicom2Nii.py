'''
05/02, at Chapel Hill
Dong

convert dicom series to nifti format
'''
import numpy
import SimpleITK as sitk
import os
from doctest import SKIP


class ScanFile(object):   
    def __init__(self,directory,prefix=None,postfix=None):  
        self.directory=directory  
        self.prefix=prefix  
        self.postfix=postfix  
          
    def scan_files(self):    
        files_list=[]    
            
        for dirpath,dirnames,filenames in os.walk(self.directory):   
            ''''' 
            dirpath is a string, the path to the directory.   
            dirnames is a list of the names of the subdirectories in dirpath (excluding '.' and '..'). 
            filenames is a list of the names of the non-directory files in dirpath. 
            '''  
            for special_file in filenames:    
                if self.postfix:    
                    special_file.endswith(self.postfix)    
                    files_list.append(os.path.join(dirpath,special_file))    
                elif self.prefix:    
                    special_file.startswith(self.prefix)  
                    files_list.append(os.path.join(dirpath,special_file))    
                else:    
                    files_list.append(os.path.join(dirpath,special_file))    
                                  
        return files_list    
      
    def scan_subdir(self):  
        subdir_list=[]  
        for dirpath,dirnames,files in os.walk(self.directory):  
            subdir_list.append(dirpath)  
        return subdir_list  
    

def main():
    path='/home/dongnie/warehouse/pelvicSeg/newData/pelvic_0118/'
    subpath='atkinson_lafayette'
    outfn=subpath+'.nii.gz'
    inputdir=path+subpath
    scan=ScanFile(path)  
    subdirs=scan.scan_subdir()  
    for subdir in subdirs:
        if subdir==path or subdir=='..':
            continue
        
        print 'subdir is, ',subdir
        
        ss=subdir.split('/')
        print 'ss is, ',ss, 'and s7 is, ',ss[7]
        
        outfn=ss[7]+'.nii.gz'
        
        reader = sitk.ImageSeriesReader()

        dicom_names = reader.GetGDCMSeriesFileNames(subdir)
        reader.SetFileNames(dicom_names)
        
        image = reader.Execute()
        
        size = image.GetSize()
        print( "Image size:", size[0], size[1], size[2] )
        
        print( "Writing image:", outfn)
        
        sitk.WriteImage(image,outfn)


if __name__ == '__main__':     
    main()
