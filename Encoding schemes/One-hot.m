
function  Binary1_code= Binary1(seq )%seq为输入的序列，Binary1为Binary-single
%A,R,N,D,C,Q,E,G,H,I,L,K,M,F,P,S,T,W,V,Y氨基酸排序
A=eye(21);%定义一个单位矩阵，即Binary编码方式
B=['A'    'R'    'N'    'D'    'C'    'Q'    'E'  ...
    'G'    'H'    'I'    'L'    'K'    'M'    'F' ...
    'P'    'S'    'T'    'W'    'V'    'Y'   'X'];%氨基酸序列对应的字母
[m,n]=size(seq);%Seq矩阵的大小
matrix_code1=[];  matrix_code2=[];  Binary1_code=[];
for i=1:m
    matrix_code2=[];%将每行的编码找到后，将matrix_code2清空用来存放下一行的编码
    %对seq按照行进行循环，找到该行每一列氨基酸对应的编码
    for j=1:n
        for k=1:21
            if seq(i,j)==B(k)
                matrix_code1=[A(k,:)];%找到序列中每个字母对应的氨基酸序列字母并编码
            end
        end
        matrix_code2=[matrix_code2, matrix_code1];%seq每行的编码
    end
    Binary1_code=[Binary1_code;matrix_code2];%seq矩阵的编码
end

end

