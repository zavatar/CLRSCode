uses cubsgp,bsgpdbg,windows,coff

loadpfm(char* src,int& tw,int& th)
	f=fopen(src,"rb")
	fscanf(f,"PF\n%d%d\n%f\n",&tw,&th,&float _1)
	pfm=new float3[tw*th]
	for(i=0;i<th;i++)
		fread(pfm+tw*(th-1-i),tw*sizeof(float3),1,f)
	fclose(f)
	ret=new float3<tw*th>
	for(i=0;i<tw*th;i++)
		ret[i]=pfm[i]
	delete pfm
	return ret

band=float3[8]const [[1,1,1],[0.8,0.8,0],[0,0.8,0.8],[0.8,0,0.8],[0,1,0],[0,0,1],[1,0,0],[0,0,0]]

t0hello()
	n=256
	forall i=0:n*n-1
		x=i%n;y=i/n
		col=$band[x*8/n];
		plot.imagexy(x,y,col);
	
t1blur()
	int w,h
	img=loadpfm("res/lena.pfm",w,h)
	r=20; A=4.f/float(r*r)
	forall i=0:w*h-1
		y=i/w;x=i-y*w
		C=img[i]
		plot.imagexy(x,y,C)
		barrier
		C_tot=make_float3(0.f,0.f,0.f)
		w_tot=0.f
		for j=-r:r
			wgt=exp(-A*float(j*j))
			w_tot+=wgt
			C_tot+=wgt*thread.get(y*w+min(max(x+j,0),w-1),C)
		C=C_tot/w_tot
		barrier
		C_tot=make_float3(0.f,0.f,0.f)
		w_tot=0.f
		for j=-r:r
			wgt=exp(-A*float(j*j))
			w_tot+=wgt
			C_tot+=wgt*thread.get(min(max(y+j,0),h-1)*w+x,C)
		C=C_tot/w_tot
		plot.imagexy(x,y,C)

struct ChosenPixel
	int2 xy
	float3 col
	
t2scan()
	int w,h
	img=loadpfm("res/lena.pfm",w,h)
	auto b
	forall i=0:w*h-1
		y=i/w;x=i-y*w
		C=img[i]
		plot.imagexy(x,y,C)
		ispurple=(C.z>(C.x+C.y)*0.6f)
		addr_purple=ispurple
		npurple=scan(rop_add,addr_purple)
		require
			b=new ChosenPixel<npurple>
		if ispurple:
			b[addr_purple].xy=make_int2(x,y)
			b[addr_purple].col=C
	forall i=0:b.n-1
		plot.imagexy(b[i].xy.x,b[i].xy.y,b[i].col)

t3sort()
	int w,h
	img=loadpfm("res/lena.pfm",w,h)
	auto b
	forall i=0:w*h-1
		y=i/w;x=i-y*w
		C=img[i]
		plot.imagexy(x,y,C)
		L=dot(C,make_float3(0.3f,0.59f,0.11f))
		thread.sortby(L)
		i=thread.rank
		y=i/w;x=i-y*w
		plot.imagexy(x,y,C)

t4median()
	int w,h
	img=loadpfm("res/lena.pfm",w,h)
	forall i=0:w*h-1
		y=i/w;x=i-y*w
		plot.imagexy(x,y,img[i])
	r=5
	forall ii=0:w*h*r*r-1
		i=ii/(r*r);j=ii-i*(r*r)
		y=i/w;x=i-y*w
		y2=j/r;x2=j-y2*r
		C=img[min(max(y+y2-r/2,0),h-1)*w+min(max(x+x2-r/2,0),w-1)]
		L=dot(C,make_float3(0.3f,0.59f,0.11f))
		static __device__ less(int2 a,int2 b){return a.x<b.x||a.x==b.x&&a.y<b.y}
		thread.sortby(make_int2(y*w+x,__float_as_int(L)),less)
		ii=thread.rank
		i=ii/(r*r);j=ii-i*(r*r)
		if j==r*r/2:
			y=i/w;x=i-y*w
			plot.imagexy(x,y,C)

t5fork()
	n=512
	forall x=0:n-1
		s=sin(float(x)/float(n)*PI*4.f)
		m=int(abs(s)*float(n)*0.5f)
		y=thread.fork(m)
		if s<0.f:y=-y
		plot.imagexy(x,y,make_float3(0.f,1.f,0.f))

t6watch()
	int w,h
	img=loadpfm("res/lena.pfm",w,h)
	r=20; A=4.f/float(r*r)
	forall i=0:w*h-1
		y=i/w;x=i-y*w
		C=img[i]
		debug.watch(C)
		barrier
		C_tot=make_float3(0.f,0.f,0.f)
		w_tot=0.f
		for j=-r:r
			wgt=exp(-A*float(j*j))
			w_tot+=wgt
			C_tot+=wgt*thread.get(y*w+min(max(x+j,0),w-1),C)
		C=C_tot/w_tot
		debug.watch(C)
		barrier
		C_tot=make_float3(0.f,0.f,0.f)
		w_tot=0.f
		for j=-r:r
			wgt=exp(-A*float(j*j))
			w_tot+=wgt
			C_tot+=wgt*thread.get(min(max(y+j,0),h-1)*w+x,C)
		C=C_tot/w_tot
		debug.watch(C)

tick()
	cuCtxSynchronize()
	QueryPerformanceCounter((void*)&i64 t0)
	return t0

t7meta()
	n=10000000
	r=new int<n>
	forall i=0:n-1
		r[i]=0
	t0=tick()
	forall i=0:n-1
		int[16] a
		for j=0:15
			a[j]=i+j
		b=0
		for j=0:i&15
			b+=a[j]
		r[i]=b
	t1=tick()
	forall i=0:n-1
		spapTargetOption('cudaLaunchMethod','$$cuda.cudevBigLoop')
		int[16] a
		For j=0:15
			a[j]=i+j
		b=0
		For j=0:15
			if j<=(i&15):
				b+=a[j]
		r[i]=b
	t2=tick()
	QueryPerformanceFrequency((void*)&i64 freq)
	writeln(double(t1-t0)/double(freq)*1000.:3:3,'ms')
	writeln(double(t2-t1)/double(freq)*1000.:3:3,'ms')

filterImage(const is_horizontal,img,w,h,r,A)
	ret=new float3<w*h>
	forall i=0:w*h-1
		y=i/w;x=i-y*w
		w_tot=0.f
		C_tot=make_float3(0.f,0.f,0.f)
		for j=-r:r
			wgt=exp(-A*float(j*j))
			w_tot+=wgt
			If is_horizontal:
				C_tot+=wgt*img[y*w+min(max(x+j,0),w-1)]
			Else
				C_tot+=wgt*img[min(max(y+j,0),h-1)*w+x]
		ret[i]=C_tot/w_tot
	return ret

t8metax()
	int w,h
	img=loadpfm("res/lena.pfm",w,h)
	forall i=0:w*h-1
		y=i/w;x=i-y*w
		C=img[i]
		plot.imagexy(x,y,C)
	r=20; A=4.f/float(r*r)
	img=filterImage(1,img,w,h,r*2,A/4.f)
	img=filterImage(0,img,w,h,r/2,A*4.f)
	forall i=0:w*h-1
		y=i/w;x=i-y*w
		C=img[i]
		plot.imagexy(x,y,C)

dlist(int) kp
int *kh(int n)
	$kp = new int<n>
	return $kp.d

__device__ int tmkill(int x)
	require
		n=0
	n=compact(kh,thread.rank,!x,target_allocator)
	spapTargetOption('cudaLaunchMethod','$$cuda.cudevBigLoop')
	barrier(RANK_REASSIGNED) require
		thread.size=n-8
	require(1)
		for i=0:$kp.n-1
			write($kp[i], ' ')
		writeln
		$kp.discard()
	ork=$kp[thread.rank]
	thread.oldrank=ork+5
	return n

tparallel()
	n = 16
	in = new int<n>
	out = new int<n>
	for i=0:n-1
		in[i] = n-i-1
		out[i] = -1
		write(in[i], ' ')
	writeln
	t0=tick()
	forall i=0:n-1
		x = in[i]
		tmkill(!x)
		out[i]=x
	t1=tick()
	for i=0:out.n-1
		write(out[i], ' ')
	writeln
	QueryPerformanceFrequency((void*)&i64 freq)
	writeln(double(t1-t0)/double(freq)*1000.:3:3,'ms')

treduce_scan()
	int ret
	n = 16*(1<<10)
	in = new int<n>
	for i=0:n-1
		in[i] = rand()&0xff
	scan(in.d,in.d,n)
	t0=tick()
	For K=0:31
		forall i=0:n-1
			spapTargetOption('cudaLaunchMethod','$$cuda.cudevBigLoop')
			x = in[i]
			//ret=reduce(rop_add, x)
			ret=scan(rop_add, x)
	t1=tick()
	QueryPerformanceFrequency((void*)&i64 freq)
	writeln(double(t1-t0)/double(freq)/32.*1000.:3:3,'ms')
	return ret

assertSort(in)
	for i=1:in.n-1
		if (in[i-1]>in[i])
			writeln(in[i-1],' ',in[i]," unsorted")
			readln
	writeln("sorted")
	
tsort()
	n = 16//(16<<20)
	in = new int<n>
	out = new int<n>
	for i=0:n-1
		in[i] = rand()
		out[i] = -1
		//write(in[i], ' ')
	//writeln
	t0=tick()
	forall i=0:n-1
		spapTargetOption('cudaLaunchMethod','$$cuda.cudevBigLoop')
		x = in[i]
		j=sort_idx(x)
		out[i]=in[j]
		//thread.sortby(x)
		//out[thread.rank] = x
	t1=tick()	
	//for i=0:n-1
		//write(in[i], ' ')
	//writeln
	//for i=0:n-1
		//write(out[i], ' ')
	//writeln
	assertSort(out)
	QueryPerformanceFrequency((void*)&i64 freq)
	writeln(double(t1-t0)/double(freq)*1000.:3:3,'ms')
	
tradixsort()
	n = (1<<20)
	in = new int<n>
	out = new int<n>
	for i=0:n-1
		in[i] = rand()
		out[i] = -1
	//	write(in[i], ' ')
	//writeln
	t0 = tick()
	forall i=0:n-1
		spapTargetOption('cudaLaunchMethod','$$cuda.cudevBigLoop')
		For j=0:sizeof(int)*8-1
			barrier
			split(in,in[i],in[i]&(1<<j),target_clear)
	t1 = tick()
	//for i=0:n-1
	//	write(in[i], ' ')
	//writeln
	assertSort(in)
	QueryPerformanceFrequency((void*)&i64 freq)
	writeln(double(t1-t0)/double(freq)*1000.:3:3,'ms')

objexport void BSGPmain()
	//void main()
	//t0hello()
	//t1blur()
	//t2scan()
	//t3sort()
	//t4median()
	//t5fork()
	//t6watch()
	//t7meta()
	//t8metax()
	//tparallel()
	//treduce_scan()
	//writeln(sum)
	tsort()
	//tradixsort()
	readln