function P=CKSAAP(seqs,A)
%导入iPTM-mLys(即所需数据)的Seq列，输入seqs=cell2mat(iPTMmLys)；
%A为矩阵，即输入所需的间隔数组成的矩阵，如A=[0 1]即输出间隔为0和间隔为1的编码，前441列对应间隔为0的编码，442-882列即对应间隔为1的编码，即可运行程序出结果
%对序列矩阵按space间隔的氨基酸对进行编码
%seqs 序列矩阵，行数为序列条数
%space 氨基酸对间隔数
amino=['A'    'R'    'N'    'D'    'C'    'Q'    'E'  ...
    'G'    'H'    'I'    'L'    'K'    'M'    'F' ...
    'P'    'S'    'T'    'W'    'V'    'Y'   'X'];%氨基酸序列对应的字母

%amino =['R'    'G'    'I'    'F'    'P'    'S'    'T'  'Y'    'V' ];
% matrix_code=zeros(length(amino),length(amino),size(seqs,1));
M=zeros(size(seqs,1),length(amino)*length(amino),size(seqs,2)-1);
for space=0:size(seqs,2)-2;
 matrix_code=zeros(length(amino),length(amino),size(seqs,1));
   for j = 1:size(seqs,1)
     seq_singal = seqs(j,:);
     for i=1:size(seqs,2)-space-1
        a1=find(amino==seq_singal(i));
        a2=find(amino==seq_singal(i+space+1));
        matrix_code(a1,a2,j)=matrix_code(a1,a2,j)+1/(size(seqs,2)-space-1);
        
     end    
   end
 for j=1:size(seqs,1)
    sub_code(j,:) = reshape(matrix_code(:,:,j)',1,length(amino)*length(amino));
 end
    M(:,:,space+1)=sub_code(:,:);
end
P=[];
for m=0:size(A,2)-1
    P=[P,M(:,:,A(1,m+1)+1)];
end
