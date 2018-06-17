function IG_code=IG(seq,K)
[m1,n1]=size(seq);
amino =['A'    'C'    'D'    'E'    'F'    'G'    'H'  'I'    'K'  'L'    'M'    'N' ...
    'P'    'Q'   'R'    'S'    'T'    'V'    'W'    'Y'   'X' ];
IGE=[];
for i=1:m1
     H=0;
     RE=0;
    for k=1:21
        posi=find( seq(i,:)==amino(k));
        len=length(posi);
        p=len/K;
        if p==0
            H=H;
            RE=RE;
        else
        H=H-p*log2(p);
        RE=RE+p*log2(p*K);
        end
        ig=H-RE;
    end
    IGE=[IGE;ig]; 
end
    IG_code= IGE;
end