function dist = chiSqrDist(H1,H2)

sum = 0;
for i=1:numel(H1)
    curr = (H1(i) + H2(i)).^2./H1(i);
    sum = sum + curr;
end

dist = sum;