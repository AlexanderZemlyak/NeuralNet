##
var expr := '3 5 2 * +'; // 3 + 5 * 2
var st := new Stack<real>;

foreach var x in expr.ToWords do
  if x.IsReal then
    st.Push(x.ToReal)
  else case x of
    '+': st.Push(st.Pop + st.Pop);
    '*': st.Push(st.Pop * st.Pop);
  end;
  
Print(st.Pop);  

