
function  Binary1_code= Binary1(seq )%seqΪ��������У�Binary1ΪBinary-single
%A,R,N,D,C,Q,E,G,H,I,L,K,M,F,P,S,T,W,V,Y����������
A=eye(21);%����һ����λ���󣬼�Binary���뷽ʽ
B=['A'    'R'    'N'    'D'    'C'    'Q'    'E'  ...
    'G'    'H'    'I'    'L'    'K'    'M'    'F' ...
    'P'    'S'    'T'    'W'    'V'    'Y'   'X'];%���������ж�Ӧ����ĸ
[m,n]=size(seq);%Seq����Ĵ�С
matrix_code1=[];  matrix_code2=[];  Binary1_code=[];
for i=1:m
    matrix_code2=[];%��ÿ�еı����ҵ��󣬽�matrix_code2������������һ�еı���
    %��seq�����н���ѭ�����ҵ�����ÿһ�а������Ӧ�ı���
    for j=1:n
        for k=1:21
            if seq(i,j)==B(k)
                matrix_code1=[A(k,:)];%�ҵ�������ÿ����ĸ��Ӧ�İ�����������ĸ������
            end
        end
        matrix_code2=[matrix_code2, matrix_code1];%seqÿ�еı���
    end
    Binary1_code=[Binary1_code;matrix_code2];%seq����ı���
end

end

