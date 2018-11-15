function  ErrorMessage=genPosNegtxtLR(newName,tmpSeq,position,LenLeft,LenRight)
if(length(position)>0)
  ModSeq=tmpSeq(position(1));
  for i=1:LenLeft
    ModSeq=['X',ModSeq];
  end   
  for i=1:LenRight
    ModSeq=[ModSeq,'X'];
  end
   %输出正类点
  fidout=fopen('pos.txt','a');
  for i=1:length(position)
     outputSeq=ModSeq;
     for j=1:LenLeft
        if(position(i)==j)
            break;
        end
        outputSeq(LenLeft+1-j)=tmpSeq(position(i)-j);
     end   
    for j=1:LenRight
        if(position(i)+j>length(tmpSeq))
            break;
        end
         outputSeq(LenLeft+1+j)=tmpSeq(position(i)+j);
    end   
    fprintf( fidout,'%s %d  %s\r\n ',newName,position(i),outputSeq); 
   end  
   fclose( fidout);
   %输出负类点
   negPosition=find(tmpSeq==ModSeq(LenLeft+1));
   fidout=fopen('neg.txt','a');
   for i=1:length(negPosition)
      if(length(find(position==negPosition(i)) )==0)
         outputSeq=ModSeq;
         for j=1:LenLeft
           if(negPosition(i)==j)
             break;
           end
         outputSeq(LenLeft+1-j)=tmpSeq(negPosition(i)-j);
         end   
         for j=1:LenRight
           if(negPosition(i)+j>length(tmpSeq))
            break;
         end
         outputSeq(LenLeft+1+j)=tmpSeq(negPosition(i)+j);
        
         end
          fprintf( fidout,'%s %d  %s\r\n ',newName,negPosition(i),outputSeq); 
    end
      
    end
     fclose( fidout);
end 
   
 







