import { NodeBuilder, FunctionNode } from 'three/webgpu';

// three caches the WGSL of every Fn().setLayout() function once per backend and reuses the
// code in every later shader. Uniforms, textures and buffers referenced inside the function are
// only declared by the shader that built it first, so any other shader using the same function
// fails to compile ("struct member ... not found"). Cache per builder and stage instead: each
// shader still gets a single copy of the function (small WGSL, fast compiles) and declares all
// of the bindings the function uses.
NodeBuilder.prototype.buildFunctionNode = function ( shaderNode ) {

	const caches = this._layoutFnCache || ( this._layoutFnCache = {} );
	const stage = this.shaderStage || 'default';
	const cache = caches[ stage ] || ( caches[ stage ] = new WeakMap() );

	let fn = cache.get( shaderNode );

	if ( fn === undefined ) {

		fn = new FunctionNode();

		const previous = this.currentFunctionNode;
		this.currentFunctionNode = fn;
		fn.code = this.buildFunctionCode( shaderNode );
		this.currentFunctionNode = previous;

		cache.set( shaderNode, fn );

	}

	return fn;

};

// Materials that supply their own velocity output (material.mrtNode with 'velocity', e.g. static
// instanced vegetation) don't need previous-frame instance matrices; three would otherwise add an
// extra vertex buffer per instanced mesh whenever the render target has a velocity output.
const _needsPreviousData = NodeBuilder.prototype.needsPreviousData;
NodeBuilder.prototype.needsPreviousData = function () {

	const m = this.material;
	if ( m && m.mrtNode && m.mrtNode.has && m.mrtNode.has( 'velocity' ) ) return false;
	return _needsPreviousData.call( this );

};
