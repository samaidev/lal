	.file	"demo.c"
	.text
	.p2align 4
	.globl	classify
	.type	classify, @function
classify:
.LFB22:
	.cfi_startproc
	movss	8(%rdi), %xmm0
	movss	12(%rdi), %xmm9
	xorl	%eax, %eax
	movss	.LC0(%rip), %xmm2
	movss	.LC1(%rip), %xmm5
	movaps	%xmm9, %xmm1
	movss	(%rdi), %xmm3
	movss	24(%rdi), %xmm8
	mulss	%xmm0, %xmm2
	movss	16(%rdi), %xmm6
	movss	28(%rdi), %xmm4
	mulss	%xmm5, %xmm1
	movaps	%xmm8, %xmm7
	mulss	%xmm5, %xmm8
	movaps	%xmm6, %xmm10
	mulss	%xmm5, %xmm10
	addss	%xmm3, %xmm2
	addss	%xmm1, %xmm2
	movss	.LC2(%rip), %xmm1
	mulss	%xmm1, %xmm7
	mulss	%xmm1, %xmm3
	mulss	%xmm0, %xmm1
	addss	%xmm7, %xmm2
	movss	.LC3(%rip), %xmm7
	mulss	%xmm0, %xmm7
	addss	%xmm7, %xmm3
	movss	.LC4(%rip), %xmm7
	mulss	%xmm7, %xmm9
	mulss	%xmm7, %xmm6
	addss	%xmm9, %xmm3
	movss	.LC5(%rip), %xmm9
	mulss	%xmm9, %xmm0
	addss	%xmm8, %xmm3
	movss	4(%rdi), %xmm8
	mulss	%xmm9, %xmm8
	comiss	%xmm2, %xmm3
	addss	%xmm8, %xmm1
	addss	%xmm8, %xmm0
	addss	%xmm10, %xmm1
	movaps	%xmm4, %xmm10
	addss	%xmm6, %xmm0
	mulss	%xmm7, %xmm10
	mulss	%xmm5, %xmm4
	addss	%xmm10, %xmm1
	addss	%xmm4, %xmm0
	jbe	.L2
	movaps	%xmm3, %xmm2
	movl	$1, %eax
.L2:
	comiss	%xmm2, %xmm1
	maxss	%xmm2, %xmm1
	movl	$2, %edx
	cmova	%edx, %eax
	ucomiss	%xmm1, %xmm0
	movl	$3, %edx
	cmova	%edx, %eax
	movl	%eax, (%rsi)
	ret
	.cfi_endproc
.LFE22:
	.size	classify, .-classify
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC6:
	.string	"usage: demo v0 v1 ... v7\n"
.LC7:
	.string	"%d\n"
	.section	.text.startup,"ax",@progbits
	.p2align 4
	.globl	main
	.type	main, @function
main:
.LFB23:
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
	jle	.L22
	movq	%rsi, %rbp
	movl	$1, %ebx
	leaq	16(%rsp), %r12
	.p2align 4
	.p2align 3
.L15:
	movq	0(%rbp,%rbx,8), %rdi
	xorl	%esi, %esi
	call	strtod@PLT
	cvtsd2ss	%xmm0, %xmm0
	movss	%xmm0, -4(%r12,%rbx,4)
	addq	$1, %rbx
	cmpq	$9, %rbx
	jne	.L15
	leaq	12(%rsp), %rsi
	movq	%r12, %rdi
	call	classify
	movl	12(%rsp), %esi
	leaq	.LC7(%rip), %rdi
	xorl	%eax, %eax
	call	printf@PLT
	xorl	%eax, %eax
.L14:
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
.L22:
	.cfi_restore_state
	movq	stderr(%rip), %rcx
	movl	$25, %edx
	movl	$1, %esi
	leaq	.LC6(%rip), %rdi
	call	fwrite@PLT
	movl	$1, %eax
	jmp	.L14
	.cfi_endproc
.LFE23:
	.size	main, .-main
	.section	.rodata.cst4,"aM",@progbits,4
	.align 4
.LC0:
	.long	1045220557
	.align 4
.LC1:
	.long	1060320051
	.align 4
.LC2:
	.long	1061997773
	.align 4
.LC3:
	.long	1050253722
	.align 4
.LC4:
	.long	1058642330
	.align 4
.LC5:
	.long	1063675494
	.ident	"GCC: (Debian 14.2.0-19) 14.2.0"
	.section	.note.GNU-stack,"",@progbits
