	.file	"demo_generic.c"
	.text
	.p2align 4
	.globl	classify_generic
	.type	classify_generic, @function
classify_generic:
.LFB24:
	.cfi_startproc
	movss	(%rdi), %xmm0
	movss	8(%rdi), %xmm2
	pxor	%xmm4, %xmm4
	movss	.LC4(%rip), %xmm9
	movss	4(%rdi), %xmm5
	shufps	$0xe0, %xmm0, %xmm0
	movq	%xmm0, %xmm0
	movss	12(%rdi), %xmm3
	movss	24(%rdi), %xmm1
	mulps	.LC9(%rip), %xmm0
	movss	.LC7(%rip), %xmm7
	mulss	%xmm9, %xmm5
	shufps	$0xe0, %xmm3, %xmm3
	movq	%xmm3, %xmm3
	shufps	$0xe0, %xmm1, %xmm1
	mulps	.LC11(%rip), %xmm3
	movq	%xmm1, %xmm1
	movss	.LC8(%rip), %xmm8
	mulps	.LC12(%rip), %xmm1
	movq	%xmm0, %xmm0
	addps	%xmm4, %xmm0
	movaps	%xmm2, %xmm4
	shufps	$0xe0, %xmm4, %xmm4
	movq	%xmm4, %xmm4
	movq	%xmm3, %xmm3
	mulps	.LC10(%rip), %xmm4
	movq	%xmm1, %xmm1
	movq	%xmm0, %xmm0
	movq	%xmm4, %xmm4
	addps	%xmm4, %xmm0
	pxor	%xmm4, %xmm4
	addss	%xmm4, %xmm5
	movss	.LC6(%rip), %xmm4
	mulss	%xmm2, %xmm4
	movq	%xmm0, %xmm0
	addps	%xmm3, %xmm0
	movss	16(%rdi), %xmm3
	mulss	%xmm9, %xmm2
	movaps	%xmm3, %xmm6
	mulss	%xmm7, %xmm6
	movq	%xmm0, %xmm0
	addss	%xmm5, %xmm4
	addps	%xmm1, %xmm0
	movss	28(%rdi), %xmm1
	mulss	%xmm8, %xmm3
	addss	%xmm5, %xmm2
	addss	%xmm4, %xmm6
	movaps	%xmm1, %xmm4
	mulss	%xmm8, %xmm4
	addss	%xmm2, %xmm3
	movaps	%xmm0, %xmm2
	shufps	$0xe5, %xmm0, %xmm0
	mulss	%xmm7, %xmm1
	comiss	%xmm2, %xmm0
	maxss	%xmm2, %xmm0
	seta	%al
	addss	%xmm6, %xmm4
	movzbl	%al, %eax
	addss	%xmm3, %xmm1
	comiss	%xmm0, %xmm4
	ja	.L6
	movaps	%xmm0, %xmm4
	movl	$3, %edx
	ucomiss	%xmm4, %xmm1
	cmova	%edx, %eax
	movl	%eax, (%rsi)
	ret
	.p2align 4,,10
	.p2align 3
.L6:
	ucomiss	%xmm4, %xmm1
	movl	$2, %eax
	movl	$3, %edx
	cmova	%edx, %eax
	movl	%eax, (%rsi)
	ret
	.cfi_endproc
.LFE24:
	.size	classify_generic, .-classify_generic
	.section	.rodata.str1.8,"aMS",@progbits,1
	.align 8
.LC13:
	.string	"usage: demo_generic v0 v1 ... v7\n"
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC14:
	.string	"%d\n"
	.section	.text.startup,"ax",@progbits
	.p2align 4
	.globl	main
	.type	main, @function
main:
.LFB25:
	.cfi_startproc
	pushq	%r12
	.cfi_def_cfa_offset 16
	.cfi_offset 12, -16
	pushq	%rbp
	.cfi_def_cfa_offset 24
	.cfi_offset 6, -24
	pushq	%rbx
	.cfi_def_cfa_offset 32
	.cfi_offset 3, -32
	subq	$48, %rsp
	.cfi_def_cfa_offset 80
	cmpl	$8, %edi
	jle	.L17
	movq	%rsi, %rbp
	movl	$1, %ebx
	leaq	16(%rsp), %r12
	.p2align 4
	.p2align 3
.L10:
	movq	0(%rbp,%rbx,8), %rdi
	xorl	%esi, %esi
	call	strtod@PLT
	cvtsd2ss	%xmm0, %xmm0
	movss	%xmm0, -4(%r12,%rbx,4)
	addq	$1, %rbx
	cmpq	$9, %rbx
	jne	.L10
	leaq	12(%rsp), %rsi
	movq	%r12, %rdi
	call	classify_generic
	movl	12(%rsp), %esi
	leaq	.LC14(%rip), %rdi
	xorl	%eax, %eax
	call	printf@PLT
	xorl	%eax, %eax
.L9:
	addq	$48, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 32
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%rbp
	.cfi_def_cfa_offset 16
	popq	%r12
	.cfi_def_cfa_offset 8
	ret
.L17:
	.cfi_restore_state
	movq	stderr(%rip), %rcx
	movl	$33, %edx
	movl	$1, %esi
	leaq	.LC13(%rip), %rdi
	call	fwrite@PLT
	movl	$1, %eax
	jmp	.L9
	.cfi_endproc
.LFE25:
	.size	main, .-main
	.section	.rodata.cst4,"aM",@progbits,4
	.align 4
.LC4:
	.long	1063675494
	.set	.LC6,.LC9+4
	.set	.LC7,.LC11
	.set	.LC8,.LC11+4
	.section	.rodata.cst16,"aM",@progbits,16
	.align 16
.LC9:
	.long	1065353216
	.long	1061997773
	.long	0
	.long	0
	.align 16
.LC10:
	.long	1045220557
	.long	1050253722
	.long	0
	.long	0
	.align 16
.LC11:
	.long	1060320051
	.long	1058642330
	.long	0
	.long	0
	.align 16
.LC12:
	.long	1061997773
	.long	1060320051
	.long	0
	.long	0
	.ident	"GCC: (Debian 14.2.0-19) 14.2.0"
	.section	.note.GNU-stack,"",@progbits
