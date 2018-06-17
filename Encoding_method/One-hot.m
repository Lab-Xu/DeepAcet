function  Binary1_code= Binary1(seq )
A=eye(21);
B=['A'    'R'    'N'    'D'    'C'    'Q'    'E'  ...
    'G'    'H'    'I'    'L'    'K'    'M'    'F' ...
    'P'    'S'    'T'    'W'    'V'    'Y'   'X'];
[m,n]=size(seq);
matrix_code1=[];  matrix_code2=[];  Binary1_code=[];
for i=1:m
    matrix_code2=[];
    for j=1:n
        for k=1:21
            if seq(i,j)==B(k)
                matrix_code1=[A(k,:)];
            end
        end
        matrix_code2=[matrix_code2, matrix_code1];
    end
    Binary1_code=[Binary1_code;matrix_code2];
end

end

