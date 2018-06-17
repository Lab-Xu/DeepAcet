function BLOSUM62_code= BLOSUM62( seq )
BLOSUM =[

     4    -1    -2    -2     0    -1    -1     0    -2    -1    -1    -1    -1    -2    -1     1     0    -3    -2     0  0
    -1     5     0    -2    -3     1     0    -2     0    -3    -2     2    -1    -3    -2    -1    -1    -3    -2    -3  0
    -2     0     6     1    -3     0     0     0     1    -3    -3     0    -2    -3    -2     1     0    -4    -2    -3  0
    -2    -2     1     6    -3     0     2    -1    -1    -3    -4    -1    -3    -3    -1     0    -1    -4    -3    -3  0
     0    -3    -3    -3     9    -3    -4    -3    -3    -1    -1    -3    -1    -2    -3    -1    -1    -2    -2    -1  0
    -1     1     0     0    -3     5     2    -2     0    -3    -2     1     0    -3    -1     0    -1    -2    -1    -2  0
    -1     0     0     2    -4     2     5    -2     0    -3    -3     1    -2    -3    -1     0    -1    -3    -2    -2  0
     0    -2     0    -1    -3    -2    -2     6    -2    -4    -4    -2    -3    -3    -2     0    -2    -2    -3    -3  0
    -2     0     1    -1    -3     0     0    -2     8    -3    -3    -1    -2    -1    -2    -1    -2    -2     2    -3  0
    -1    -3    -3    -3    -1    -3    -3    -4    -3     4     2    -3     1     0    -3    -2    -1    -3    -1     3  0
    -1    -2    -3    -4    -1    -2    -3    -4    -3     2     4    -2     2     0    -3    -2    -1    -2    -1     1  0
    -1     2     0    -1    -3     1     1    -2    -1    -3    -2     5    -1    -3    -1     0    -1    -3    -2    -2  0
    -1    -1    -2    -3    -1     0    -2    -3    -2     1     2    -1     5     0    -2    -1    -1    -1    -1     1  0
    -2    -3    -3    -3    -2    -3    -3    -3    -1     0     0    -3     0     6    -4    -2    -2     1     3    -1  0
    -1    -2    -2    -1    -3    -1    -1    -2    -2    -3    -3    -1    -2    -4     7    -1    -1    -4    -3    -2  0
     1    -1     1     0    -1     0     0     0    -1    -2    -2     0    -1    -2    -1     4     1    -3    -2    -2  0
     0    -1     0    -1    -1    -1    -1    -2    -2    -1    -1    -1    -1    -2    -1     1     5    -2    -2     0  0
    -3    -3    -4    -4    -2    -2    -3    -2    -2    -3    -2    -3    -1     1    -4    -3    -2    11     2    -3  0
    -2    -2    -2    -3    -2    -1    -2    -3     2    -1    -1    -2    -1     3    -3    -2    -2     2     7    -1  0
     0    -3    -3    -3    -1    -2    -2    -3    -3     3     1    -2     1    -1    -2    -2     0    -3    -1     4  0
     0    0     0      0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0  1];
 
 B=['A'    'R'    'N'    'D'    'C'    'Q'    'E'  ...
    'G'    'H'    'I'    'L'    'K'    'M'    'F' ...
    'P'    'S'    'T'    'W'    'V'    'Y'   'X'];
    [m,n]=size(seq);
    matrix_code1=[];  matrix_code2=[];  BLOSUM62_code=[];
    for i=1:m
        matrix_code2=[];
        for j=1:n
            for k=1:21
                if seq(i,j)==B(k)
                    matrix_code1=[BLOSUM(k,:)];
                end
            end
            matrix_code2=[matrix_code2, matrix_code1];
       BLOSUM62_code=[BLOSUM62_code;matrix_code2];
    end
    
end

