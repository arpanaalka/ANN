A1 = [-1 -1 -1 -1 -1 1 1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 1 1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 1 1 -1 -1 -1 -1 -1 -1 -1 -1 -1 1 1 1 1 -1 -1 -1 -1 -1 -1 -1 1 1 1 1 1 1 -1 -1 -1 -1 -1 1 1 1 1 1 1 1 1 -1 -1 -1 1 1 1 -1 1 1 -1 1 1 1 -1 1 1 1 -1 -1 1 1 -1 -1 1 1 1 1 1 -1 -1 -1 1 1 -1 -1 -1 1 1 1 -1 -1 -1 -1 1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 1 1 1 -1 -1 -1 -1 -1 -1 -1 -1 1 -1 -1 1 -1 -1 -1 -1];
  size(A1)
  C1 = [0 0 0 1 1 1 1 0 0 0;
0 0 0 1 1 1 1 0 0 0;
0 0 0 1 1 1 1 0 0 0;
0 0 0 1 1 1 1 0 0 0;
0 0 0 1 1 1 1 0 0 0;
0 0 0 1 1 1 1 0 0 0;
0 0 0 1 1 1 1 0 0 0;
0 0 0 1 1 1 1 0 0 0;
0 0 0 1 1 1 1 0 0 0;
0 0 0 1 1 1 1 0 0 0;
0 0 0 1 1 1 1 0 0 0;
0 0 0 1 1 1 1 0 0 0];
  
B1 = [-1 -1 -1 1 1 1 1 -1 -1 -1 -1 -1 -1 1 1 1 1 -1 -1 -1 -1 -1 -1 1 1 1 1 -1 -1 -1 -1 -1 -1 1 1 1 1 -1 -1 -1 -1 -1 -1 1 1 1 1 -1 -1 -1 -1 -1 -1 1 1 1 1 -1 -1 -1 -1 -1 -1 1 1 1 1 -1 -1 -1 -1 -1 -1 1 1 1 1 -1 -1 -1 -1 -1 -1 1 1 1 1 -1 -1 -1 -1 -1 -1 1 1 1 1 -1 -1 -1 -1 -1 -1 1 1 1 1 -1 -1 -1 -1 -1 -1 1 1 1 1 -1 -1 -1];
A2 = [-1 -1 -1 -1 1 1 1 1 -1 -1 -1 -1 -1 -1 -1 -1 -1 1 1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 1 1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 1 1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 1 1 -1 -1 -1 -1 -1 -1 1 1 -1 -1 1 1 -1 -1 1 1 -1  -1 1 1 -1 1 1 1 1 -1 1 1 -1 -1 1 1 1 1 1 1 1 1 1 1 -1 -1 1 1 1 1 1 1 1 1 1 1 -1 -1 1 1 -1 1 1 1 1 -1 1 1 -1 -1 1 1 -1 -1 1 1 -1 -1 1 1 -1 -1 1 1 -1 -1 -1 -1 -1 -1 1 1 -1];
  size(A2)
  C2 = [1 1 1 1 1 1 1 1 0 0;
1 1 1 1 1 1 1 1 0 0;
0 0 0 0 0 0 1 1 0 0;
0 0 0 0 0 0 1 1 0 0; 
0 0 0 0 0 0 1 1 0 0;
1 1 1 1 1 1 1 1 0 0;
1 1 1 1 1 1 1 1 0 0; 
1 1 0 0 0 0 0 0 0 0; 
1 1 0 0 0 0 0 0 0 0;
1 1 0 0 0 0 0 0 0 0;
1 1 1 1 1 1 1 1 0 0;
1 1 1 1 1 1 1 1 0 0];
  
B2 = [1 1 1 1 1 1 1 1 -1 -1 1 1 1 1 1 1 1 1 -1 -1 -1 -1 -1 -1 -1 -1 1 1 -1 -1 -1 -1 -1 -1 -1 -1 1 1 -1 -1 -1 -1 -1 -1 -1 -1 1 1 -1 -1 1 1 1 1 1 1 1 1 -1 -1 1 1 1 1 1 1 1 1 -1 -1 1 1 -1 -1 -1 -1 -1 -1 -1 -1 1 1 -1 -1 -1 -1 -1 -1 -1 -1 1 1 -1 -1 -1 -1 -1 -1 -1 -1 1 1 1 1 1 1 1 1 -1 -1 1 1 1 1 1 1 1 1 -1 -1];
A3 = [-1 -1 -1 -1 -1 1 1 -1 -1 -1 -1 -1 -1 -1 -1 -1 1 1 1 1 -1 -1 -1 -1 -1 -1 -1 -1 -1 1 1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 1 1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 1 1 -1 -1 -1 -1 -1 -1 1 -1 -1 -1 1 1 -1 -1 -1 1 -1 -1 -1 1 -1 -1 1 1 -1 -1 1 -1 -1 -1 -1 -1 1 1 1 1 1 1 -1 -1 -1 -1 -1 1 1 1 1 1 1 1 1 -1 -1 -1 -1 -1 1 1 1 1 1 1 -1 -1 -1 -1 -1 1 -1 -1 1 1 -1 -1 1 -1 -1 -1 1 -1 -1 -1 -1 -1 -1 -1 -1 1 -1];
  size(A3)
  
B3 = [-1 -1 1 1 1 1 1 1 -1 -1 -1 -1 1 1 1 1 1 1 1 -1 -1 -1 -1 -1 -1 -1 -1 1 1 -1 -1 -1 -1 -1 -1 -1 -1 1 1 -1 -1 -1 -1 -1 -1 -1 -1 1 1 -1 -1 -1 -1 -1 1 1 1 1 -1 -1 -1 -1 -1 -1 1 1 1 1 -1 -1 -1 -1 -1 -1 -1 -1 -1 1 1 -1 -1 -1 -1 -1 -1 -1 -1 1 1 -1 -1 -1 -1 -1 -1 -1 -1 1 1 -1 -1 -1 1 1 1 1 1 1 -1 -1 -1 -1 1 1 1 1 1 1 -1 -1]; 
size(B3)

  C3 = [0 0 1 1 1 1 1 1 0 0;
0 0 1 1 1 1 1 1 1 0;
0 0 0 0 0 0 0 1 1 0;
0 0 0 0 0 0 0 1 1 0;
0 0 0 0 0 0 0 1 1 0;
0 0 0 0 1 1 1 1 0 0;
0 0 0 0 1 1 1 1 0 0;
0 0 0 0 0 0 0 1 1 0;
0 0 0 0 0 0 0 1 1 0;
0 0 0 0 0 0 0 1 1 0;
0 0 1 1 1 1 1 1 0 0;
0 0 1 1 1 1 1 1 0 0];

M = A1'*B1 + A2'*B2 +A3'*B3;
size(M)


Anew = [1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 1 -1 -1 -1 -1 -1 -1 1 1 -1 -1 -1 1 -1 -1 -1 -1 -1 1 -1 -1 -1 1 1 -1 1 -1 -1 1 -1 -1 1 -1 1 -1 -1 -1 1 -1 1 -1 -1 -1 -1 1 1 -1 -1 1 1 1 1 -1 -1 -1 1 1 1 -1 1 1 -1 -1 1 1 -1 -1 1 -1 -1 -1 1 -1 -1 1 1 1 1 1 -1 1 1 1 1 1 -1 1 1 1 -1 1 1 1 -1 1 1 -1 -1 -1 -1 -1 1 1 -1 -1 -1 1 1 -1 1 1 -1 -1 1 -1 1 -1 1 1 -1 1 -1 -1 -1 -1 -1];

alpha = zeros(1,144);
beta = zeros(1,120);

for i = 1 : 10
    beta = Anew*M;
    for i = 1 : 120
        if beta(1,i) > 0
            beta(1,i) = 1;
        elseif beta(1,i) < 0
            beta(1,i) = -1;
        end
    end
    
    if beta == B1
        disp('Its one')
        disp(C1)
        break;
    elseif beta == B1
        disp('Its Two')
        disp(C2)
        break;
    elseif beta == B3
        disp('Its Three')
        disp(C3)
        break;
    else
        Anew = beta*M';
        for i = 1 : 144
            if Anew(1,i) > 0
               Anew(1,i) = 1;
            elseif Anew(1,i) < 0
               Anew(1,i) = -1;
            end
        end
    end
end