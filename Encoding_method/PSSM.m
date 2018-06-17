function  [PSSM_pos,PSSM_neg]=PSSM( pos_seq,neg_seq)
[m1,n1]=size(pos_seq);
[m2,n2]=size(neg_seq);
amino=['A'    'R'    'N'    'D'    'C'    'Q'    'E'  ...
    'G'    'H'    'I'    'L'    'K'    'M'    'F' ...
    'P'    'S'    'T'    'W'    'V'    'Y'   'X'];
[m3,n3]=size(amino);
POS=zeros(n1*n3,m1);
NEG=zeros(n2*n3,m2);
for i=1:n3
     for j=1:n1
      for k=1:m1
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
Pvalues=[];
for len=1:size(POS,1)
    DataX=POS(len,:);
    DataY=NEG(len,:);
    pvalues = mattest(DataX, DataY);
    Pvalues=[Pvalues,pvalues];
end
PV=reshape(Pvalues,n1,n3)';
Code1=[];
for id1=1:n1
    D1=[];
    for k=1:21
        posi=find( pos_seq(:,id1)==amino(k));
        num=length(posi);
        frequency=num/m1;
        D1=[D1; frequency];
    end
    Code1=[Code1,D1];
end
pos_code=Code1;
Code2=[];
for id2=1:n2
    D2=[];
    for k=1:21
        posi=find( neg_seq(:,id2)==amino(k));%
        num=length(posi);
        frequency=num/m2;
        D2=[D2; frequency];
    end
    Code2=[Code2,D2];
end
neg_code=Code2;

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

PSSM_Code=reshape(PSSM_code,n1,n3)';
PSSM_pos=[];PSSM_pos0=[];PSSM_pos1=[];
 for i=1:size(pos_seq,1)
        PSSM_pos1=[];
        for j=1:size(pos_seq,2)
            for k=1:n3
                if pos_seq(i,j)==amino(k)
                     PSSM_pos0=[PSSM_Code(k,j)];
                end
            end
            PSSM_pos1=[PSSM_pos1, PSSM_pos0];
        end
      PSSM_pos=[PSSM_pos;PSSM_pos1];
    end
PSSM_neg=[];PSSM_neg0=[];PSSM_neg1=[];
 for i=1:size(neg_seq,1)
        PSSM_neg1=[];
        for j=1:size(neg_seq,2)
            for k=1:n3
                if neg_seq(i,j)==amino(k)
                    PSSM_neg0=[PSSM_Code(k,j)];
                end
            end
            PSSM_neg1=[PSSM_neg1, PSSM_neg0];
        end
      PSSM_neg=[PSSM_neg;PSSM_neg1];
  end
%

    




