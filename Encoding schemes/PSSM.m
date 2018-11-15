function  [PSSM_pos,PSSM_neg]=PSSM( pos_seq,neg_seq)%pos_seq为正类，neg_seq为负类
%实验数据
% pos_seq=Homo_pos15_Seq;
% neg_seq=Homo_neg15_Seq;
[m1,n1]=size(pos_seq);
[m2,n2]=size(neg_seq);
amino=['A'    'R'    'N'    'D'    'C'    'Q'    'E'  ...
    'G'    'H'    'I'    'L'    'K'    'M'    'F' ...
    'P'    'S'    'T'    'W'    'V'    'Y'   'X'];%氨基酸序列对应的字母
[m3,n3]=size(amino);
POS=zeros(n1*n3,m1);
NEG=zeros(n2*n3,m2);
%将矩阵中每一列每个位置中出氨基酸（按照氨基酸序列循环）记为1
for i=1:n3%按照字母循环
     for j=1:n1%按照列循环
      for k=1:m1%行循环
       if pos_seq(k,j)==amino(i)
           POS((i-1)*n1+j,k)=1;
       else
           POS((i-1)*n1+j,k)=0;
      end
     end
     end   
end
for i=1:n3
     for j=1:n2
      for k=1:m2
       if neg_seq(k,j)==amino(i)
           NEG((i-1)*n2+j,k)=1;
       else
           NEG((i-1)*n2+j,k)=0;
      end
     end
     end   
end
%将正负类得到的编码矩阵按照行进t检验
Pvalues=[];
for len=1:size(POS,1)
    DataX=POS(len,:);
    DataY=NEG(len,:);
    pvalues = mattest(DataX, DataY);
    Pvalues=[Pvalues,pvalues];
end
PV=reshape(Pvalues,n1,n3)';%将其变为21*31的矩阵
Code1=[];
for id1=1:n1%对pos_seq进行列循环
    D1=[];
    for k=1:21
        posi=find( pos_seq(:,id1)==amino(k));%找到每列中的氨基酸对应B矩阵中的氨基酸的位置
        num=length(posi);%每列中每种氨基酸的个数
        frequency=num/m1;%每种氨基酸在该列所占的比例
        D1=[D1; frequency];%每列每种氨基酸的比例
    end
    Code1=[Code1,D1];%pos矩阵中每种氨基酸在该列所占的比例，每行按照氨基酸序列排列（即第一行对应A,第二行对应R,以此类推）
end
pos_code=Code1;
Code2=[];
for id2=1:n2%对neg_seq进行列循环
    D2=[];
    for k=1:21
        posi=find( neg_seq(:,id2)==amino(k));%找到每列中的氨基酸对应B矩阵中的氨基酸的位置
        num=length(posi);%每列中每种氨基酸的个数
        frequency=num/m2;%每种氨基酸在该列所占的比例
        D2=[D2; frequency];%每列每种氨基酸的比例
    end
    Code2=[Code2,D2];%pos矩阵中每种氨基酸在该列所占的比例，每行按照氨基酸序列排列（即第一行对应A,第二行对应R,以此类推）
end
neg_code=Code2;

%求PSSM矩阵
PSSM_code=[];
for p=1:size(PV,1)
    for q=1:size(PV,2)
        theda=(pos_code(p,q)-neg_code(p,q))/PV(p,q);
        if theda>=0
            pssm=log(abs(theda)+1);
        else
            pssm=-log(abs(theda)+1);
        end
        PSSM_code=[PSSM_code,pssm];
            
    end
        
end
% posi=find(isnan(PSSM_code)==1);%找到值为NaN的元素
% PSSM_code(posi)=0;%将NaN改为0
PSSM_Code=reshape(PSSM_code,n1,n3)';%将其改为n3*n1的矩阵,得到PSSM矩阵
PSSM_pos=[];PSSM_pos0=[];PSSM_pos1=[];
 for i=1:size(pos_seq,1)
        PSSM_pos1=[];%将每行的编码找到后，将 PSSM_pos1清空用来存放下一行的编码
        %对pos_seq按照行进行循环，找到该行每一列氨基酸对应的编码
        for j=1:size(pos_seq,2)
            for k=1:n3
                if pos_seq(i,j)==amino(k)
                     PSSM_pos0=[PSSM_Code(k,j)];%找到序列中每个字母对应的氨基酸序列字母并编码
                end
            end
            PSSM_pos1=[PSSM_pos1, PSSM_pos0];%pos_seq每行的编码
        end
      PSSM_pos=[PSSM_pos;PSSM_pos1];%pos_seq矩阵的编码
    end
PSSM_neg=[];PSSM_neg0=[];PSSM_neg1=[];
 for i=1:size(neg_seq,1)
        PSSM_neg1=[];%将每行的编码找到后，将 PSSM_neg1清空用来存放下一行的编码
        %对neg_seq按照行进行循环，找到该行每一列氨基酸对应的编码
        for j=1:size(neg_seq,2)
            for k=1:n3
                if neg_seq(i,j)==amino(k)
                    PSSM_neg0=[PSSM_Code(k,j)];%找到序列中每个字母对应的氨基酸序列字母并编码
                end
            end
            PSSM_neg1=[PSSM_neg1, PSSM_neg0];%neg_seq每行的编码
        end
      PSSM_neg=[PSSM_neg;PSSM_neg1];%neg_seq矩阵的编码
  end
%

    




