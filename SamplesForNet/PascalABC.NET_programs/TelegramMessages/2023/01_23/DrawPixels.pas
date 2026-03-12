##
uses GraphWPF;

DrawPixels(0,0,MatrGen(800,600,(ix,iy)->GrayColor(255-SeqWhile(Cplx(0,0),z -> z * z + 0.0035 * Cplx(ix - 600,iy - 300), z -> z.Magnitude<10).Take(255).Count)));
