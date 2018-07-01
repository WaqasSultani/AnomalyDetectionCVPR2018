clc
clear all
close all

ROC_PathAL='/Users/Waqas/Desktop/Presentation/PaperAL/CVPR2018/Code/PublicCode/GithubCode/Paper_Results';
All_files=dir([ROC_PathAL,'/*.mat']);
%All_files=All_files(3:end);
Colors={'b','c','k','r'};

AUC_All=[];
for i=1:length(All_files) 
     
    
    FilePath=[ROC_PathAL,'/',All_files(i).name]
    load(FilePath)
    
    plot(X,Y,'Color',Colors{i},'LineWidth',3.5);
    hold on;
    AUC_All=[AUC_All;AUC]
    clear X  Y
    
end

AUC_All*100

legend({'Binary classifier','Lu et al.','Hassan et al.','Proposed with constraints'},'FontSize',16,'Location','southeast');
xlabel('False Positive Rate','FontWeight','normal','FontSize',18);
ylabel('True Positive Rate','FontWeight','normal','FontSize',18);
set(gca,'FontWeight','normal','FontSize',12);

grid on



 
