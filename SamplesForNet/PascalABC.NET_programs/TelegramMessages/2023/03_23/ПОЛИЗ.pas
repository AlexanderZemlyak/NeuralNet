##
var expr := '1 2 + 4 * 3 +';
var ss := expr.Split();
var st := new Stack<integer>();

foreach var x in ss do
    if x[1].IsDigit then
        st.Push(x.ToInteger)
    else if x = '+' then
        st.Push(st.Pop + st.Pop)
    else if x = '*' then
        st.Push(st.Pop * st.Pop);

Println(st.Pop);