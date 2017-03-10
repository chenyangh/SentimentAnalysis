y = [];
c = categorical({'1','2','3','4','5'});
for i = 1:5
    y = [y ;a(i) b(i)];
end

bar(y)