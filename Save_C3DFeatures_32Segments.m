clc
clear all
close all

% This code save already computed C3D features into 32 (video features) segments.
% We assume that C3D features for a video are already computed. We use
% default settings for computing C3D features, i.e., we compute C3D features for
% every 16 frames and obtain the features from fc6.
 
C3D_Path='/newdata/UCF_Anomaly_Dataset/Dataset_iA/PRIA_Data/Dataset/C3D_Features';
C3D_Path_Seg='/newdata/UCF_Anomaly_Dataset/Dataset_iA/PRIA_Data/Dataset/C3D_Features_txt/Avg';

 
if ~exist(C3D_Path_Seg,'dir')
     mkdir(C3D_Path_Seg)
end
 

All_Folder=dir(C3D_Path);
All_Folder=All_Folder(3:end);
subcript='_C.txt';

for ifolder=1:length(All_Folder)
     
    Folder_Path=[C3D_Path,'/',All_Folder(ifolder).name];
    %Folder_Path is path of a folder which contains C3D features (for every 16 frames) for a paricular video.
    
    
    AllFiles=dir([Folder_Path,'/*.fc6-1']);
    Feature_vect=zeros(length(AllFiles),4096);
     for ifile=1:length(AllFiles)
          FilePath=[Folder_Path,'/', AllFiles(ifile).name];  
          [s, data] = read_binary_blob(FilePath);
          Feature_vect(ifile,:)=data;
          clear data
     end
     
      if sum( Feature_vect(:))==0
           error('??')
      end
      
      
      % Write C3D features in text file to load in
      % Training_AnomalyDetector_public ( You can directly use .mat format if you want).
      
       fid1=fopen([C3D_Path_Seg,'/',All_Folder(ifolder).name,subcript],'w'); 
       if ~isempty(find(sum(Feature_vect,2)==0))
             error('??')
       end
  
     
      if ~isempty(find(isnan(Feature_vect(:))))
             error('??')
      end
    
      if ~isempty(find(Feature_vect(:)==Inf))
             error('??')
      end
      
    
   
  %% 32 Segments
     
     Segments_Features=zeros(32,4096);
     thirty2_shots= round(linspace(1,length(AllFiles),33));
     count=0;
     for ishots=1:length(thirty2_shots)-1
        
        ss=thirty2_shots(ishots);
        ee=thirty2_shots(ishots+1)-1;
        if ishots==length(thirty2_shots)
            ee=thirty2_shots(ishots+1);
         end
        
        if ss==ee
        
            temp_vect=Feature_vect(ss:ee,:);
            
        elseif ee<ss
           
           temp_vect=Feature_vect(ss,:);
            
        else
            temp_vect=mean(Feature_vect(ss:ee,:));
        end
        
        
        temp_vect=temp_vect/norm(temp_vect);
        
        if norm(temp_vect)==0
           error('??')
        end
        
        
        count=count+1;
        Segments_Features(count,:)=temp_vect;
        
        
     end
    
  %verify
  
      if ~isempty(find(sum(Segments_Features,2)==0))
             error('??')
      end
  
     
      if ~isempty(find(isnan(Segments_Features(:))))
             error('??')
      end
    
      if ~isempty(find(Segments_Features(:)==Inf))
             error('??')
      end
      
 % save 32 segment features in text file ( You can directly save and load .mat file in python as well).
 
     for ii=1:size(Segments_Features,1)
         feat_text=Segments_Features(ii,:);%(Feature_vect(ii,:));
         fprintf (fid1,'%f ',feat_text);
         fprintf (fid1,'\n'); 
     end
     
  fclose(fid1);
  
  
  
end
    

    
