function  [PSSM_pos,PSSM_neg]=PSSM( pos_seq,neg_seq)%pos_seqΪ���࣬neg_seqΪ����
%ʵ������
% pos_seq=Homo_pos15_Seq;
% neg_seq=Homo_neg15_Seq;
[m1,n1]=size(pos_seq);
[m2,n2]=size(neg_seq);
amino=['A'    'R'    'N'    'D'    'C'    'Q'    'E'  ...
    'G'    'H'    'I'    'L'    'K'    'M'    'F' ...
    'P'    'S'    'T'    'W'    'V'    'Y'   'X'];%���������ж�Ӧ����ĸ
[m3,n3]=size(amino);
POS=zeros(n1*n3,m1);
NEG=zeros(n2*n3,m2);
%��������ÿһ��ÿ��λ���г������ᣨ���հ���������ѭ������Ϊ1
for i=1:n3%������ĸѭ��
     for j=1:n1%������ѭ��
      for k=1:m1%��ѭ��
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
%��������õ��ı���������н�t����
Pvalues=[];
for len=1:size(POS,1)
    DataX=POS(len,:);
    DataY=NEG(len,:);
    pvalues = mattest(DataX, DataY);
    Pvalues=[Pvalues,pvalues];
end
PV=reshape(Pvalues,n1,n3)';%�����Ϊ21*31�ľ���
Code1=[];
for id1=1:n1%��pos_seq������ѭ��
    D1=[];
    for k=1:21
        posi=find( pos_seq(:,id1)==amino(k));%�ҵ�ÿ���еİ������ӦB�����еİ������λ��
        num=length(posi);%ÿ����ÿ�ְ�����ĸ���
        frequency=num/m1;%ÿ�ְ������ڸ�����ռ�ı���
        D1=[D1; frequency];%ÿ��ÿ�ְ�����ı���
    end
    Code1=[Code1,D1];%pos������ÿ�ְ������ڸ�����ռ�ı�����ÿ�а��հ������������У�����һ�ж�ӦA,�ڶ��ж�ӦR,�Դ����ƣ�
end
pos_code=Code1;
Code2=[];
for id2=1:n2%��neg_seq������ѭ��
    D2=[];
    for k=1:21
        posi=find( neg_seq(:,id2)==amino(k));%�ҵ�ÿ���еİ������ӦB�����еİ������λ��
        num=length(posi);%ÿ����ÿ�ְ�����ĸ���
        frequency=num/m2;%ÿ�ְ������ڸ�����ռ�ı���
        D2=[D2; frequency];%ÿ��ÿ�ְ�����ı���
    end
    Code2=[Code2,D2];%pos������ÿ�ְ������ڸ�����ռ�ı�����ÿ�а��հ������������У�����һ�ж�ӦA,�ڶ��ж�ӦR,�Դ����ƣ�
end
neg_code=Code2;

%��PSSM����
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
% posi=find(isnan(PSSM_code)==1);%�ҵ�ֵΪNaN��Ԫ��
% PSSM_code(posi)=0;%��NaN��Ϊ0
PSSM_Code=reshape(PSSM_code,n1,n3)';%�����Ϊn3*n1�ľ���,�õ�PSSM����
PSSM_pos=[];PSSM_pos0=[];PSSM_pos1=[];
 for i=1:size(pos_seq,1)
        PSSM_pos1=[];%��ÿ�еı����ҵ��󣬽� PSSM_pos1������������һ�еı���
        %��pos_seq�����н���ѭ�����ҵ�����ÿһ�а������Ӧ�ı���
        for j=1:size(pos_seq,2)
            for k=1:n3
                if pos_seq(i,j)==amino(k)
                     PSSM_pos0=[PSSM_Code(k,j)];%�ҵ�������ÿ����ĸ��Ӧ�İ�����������ĸ������
                end
            end
            PSSM_pos1=[PSSM_pos1, PSSM_pos0];%pos_seqÿ�еı���
        end
      PSSM_pos=[PSSM_pos;PSSM_pos1];%pos_seq����ı���
    end
PSSM_neg=[];PSSM_neg0=[];PSSM_neg1=[];
 for i=1:size(neg_seq,1)
        PSSM_neg1=[];%��ÿ�еı����ҵ��󣬽� PSSM_neg1������������һ�еı���
        %��neg_seq�����н���ѭ�����ҵ�����ÿһ�а������Ӧ�ı���
        for j=1:size(neg_seq,2)
            for k=1:n3
                if neg_seq(i,j)==amino(k)
                    PSSM_neg0=[PSSM_Code(k,j)];%�ҵ�������ÿ����ĸ��Ӧ�İ�����������ĸ������
                end
            end
            PSSM_neg1=[PSSM_neg1, PSSM_neg0];%neg_seqÿ�еı���
        end
      PSSM_neg=[PSSM_neg;PSSM_neg1];%neg_seq����ı���
  end
%

    




