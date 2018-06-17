function P=CKSAAP(seqs,A)
%����iPTM-mLys(����������)��Seq�У�����seqs=cell2mat(iPTMmLys)��
%AΪ���󣬼���������ļ������ɵľ�����A=[0 1]��������Ϊ0�ͼ��Ϊ1�ı��룬ǰ441�ж�Ӧ���Ϊ0�ı��룬442-882�м���Ӧ���Ϊ1�ı��룬�������г�������
%�����о���space����İ�����Խ��б���
%seqs ���о�������Ϊ��������
%space ������Լ����
amino=['A'    'R'    'N'    'D'    'C'    'Q'    'E'  ...
    'G'    'H'    'I'    'L'    'K'    'M'    'F' ...
    'P'    'S'    'T'    'W'    'V'    'Y'   'X'];%���������ж�Ӧ����ĸ

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
