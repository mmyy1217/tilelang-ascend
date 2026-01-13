
static llvm::SmallVector<int64_t>
getBroadcastDim(const Array<PrimExpr> &buffer_shape0,
                const std::vector<int64_t> &buffer_shape1) {
  llvm::SmallVector<int64_t> dims;

  if (buffer_shape0.empty() || buffer_shape1.empty()) {
    return dims;
  }

  int64_t rank0 = buffer_shape0.size();
  int64_t rank1 = buffer_shape1.size();
  int64_t outRank = std::max(rank0, rank1);

  // i: 输出维度索引（从左到右）
  for (int64_t i = 0; i < outRank; ++i) {
    // 对应到 input 的 index（右对齐）
    int64_t idx0 = i - (outRank - rank0);
    int64_t idx1 = i - (outRank - rank1);

    int64_t dim0 = 1;
    int64_t dim1 = 1;

    if (idx0 >= 0) {
      const int64_t* v0 = as_const_int(buffer_shape0[idx0]);
      CHECK(v0) << "buffer_shape0 must be constant int";
      dim0 = *v0;
    }

    if (idx1 >= 0) {
      dim1 = buffer_shape1[idx1];
    }

    if (dim0 == 1 && dim1 != 1) {
      dims.emplace_back(i);
    } else if (dim0 != 1 && dim1 == 1) {
      dims.emplace_back(i);
    } else {
      CHECK(dim0 == dim1)
          << "Incompatible broadcast at axis " << i
          << ": " << dim0 << " vs " << dim1;
    }
  }

  return dims;
}

void CodeGenTileLangNPUIRDEV::VselectCodegen(const CallNode *op) {
  /// Generate hivm.hir.vsel for tl.npuir_select.
  /// before:
  ///   T.npuir_select(Cond_VEC, A_VEC, B_VEC, C_VEC)
  /// after:
  ///  %8 = tensor.empty() : tensor<32xf16>
  ///  %9 = hivm.hir.vsel ins(%Cond_VEC, %A_VEC, %B_VEC : tensor<32xi1>, tensor<32xf16>, tensor<32xf16>) outs(%8 : tensor<32xf16>) -> tensor<32xf16>
  ///  %c1 = arith.constant 1 : index
  ///  %c32 = arith.constant 32 : index
  ///  %from_elements = tensor.from_elements %c1, %c32 : tensor<2xindex>
  ///  %reshape = tensor.reshape %9(%from_elements) : (tensor<32xf16>, tensor<2xindex>) -> tensor<1x32xf16>
  ///  %inserted_slice = tensor.insert_slice %reshape into %C_VEC[%7, 0] [1, 32] [1, 1] : tensor<1x32xf16> into tensor<8x32xf16>

  tvm::tl::NpuirSelect npuirop(op->args, this->vmap);

    // Retrieve offests, sizes, and strides from Range
  auto createOpFoldResultArray = [&](const Array<Range>& range) 
      -> std::tuple<SmallVector<OpFoldResult>, 
                    SmallVector<OpFoldResult>, 
                    SmallVector<OpFoldResult>> {
    SmallVector<OpFoldResult> offsets;
    SmallVector<OpFoldResult> sizes;
    SmallVector<OpFoldResult> strides;
    for (const auto& r : range) {
      // offset
      if (auto offset_int = as_const_int(r->min)) {
        offsets.push_back(builder.getI64IntegerAttr(*offset_int));
      } else {
        mlir::Value offsetVal = CreateIndexCastOp(MakeValue(r->min));
        offsets.push_back(offsetVal);
      }
      // size
      if (auto size_int = as_const_int(r->extent)) {
        sizes.push_back(builder.getI64IntegerAttr(*size_int));
      } else {
        mlir::Value sizeVal = CreateIndexCastOp(MakeValue(r->extent));
        sizes.push_back(sizeVal);
      }
      // stride (usually is 1)
      strides.push_back(builder.getI64IntegerAttr(1));
      }
      return {offsets, sizes, strides};
    };

    auto createCastIfTypeMismatch = [&](mlir::Value src_value, mlir::Value dst_value) -> mlir::Value {
    auto src_type = src_value.getType();
    auto dst_type = dst_value.getType();
    
    // Get src and dst ElementType
    mlir::Type src_element_type, dst_element_type;
    if (auto src_tensor_type = src_type.dyn_cast<mlir::TensorType>()) {
      src_element_type = src_tensor_type.getElementType();
    } else if (auto src_memref_type = src_type.dyn_cast<mlir::MemRefType>()) {
      src_element_type = src_memref_type.getElementType();
    } else {
      return src_value;
    }
    if (auto dst_tensor_type = dst_type.dyn_cast<mlir::TensorType>()) {
      dst_element_type = dst_tensor_type.getElementType();
    } else if (auto dst_memref_type = dst_type.dyn_cast<mlir::MemRefType>()) {
      dst_element_type = dst_memref_type.getElementType();
    } else {
      return src_value;
    }
    // No cast if ElementType are the same
    if (src_element_type == dst_element_type) {
      return src_value;
    }
    
    // Get src tensor shape
    llvm::ArrayRef<int64_t> src_shape;
    if (auto src_tensor_type = src_type.dyn_cast<mlir::RankedTensorType>()) {
      src_shape = src_tensor_type.getShape();
    } else if (auto src_memref_type = src_type.dyn_cast<mlir::MemRefType>()) {
      src_shape = src_memref_type.getShape();
    } else {
      return src_value;
    }
    
    // Create VCastOp
    auto castDstTensor = builder.create<mlir::tensor::EmptyOp>(
        builder.getUnknownLoc(), src_shape, dst_element_type);
    mlir::Type dst_type_ = castDstTensor.getType();
    mlir::TypeRange result_tensors(&dst_type_, 1);
    mlir::hivm::RoundMode mode = mlir::hivm::RoundMode::RINT;
    auto newCastOp = builder.create<mlir::hivm::VCastOp>(
        builder.getUnknownLoc(), result_tensors, src_value, 
        castDstTensor.getResult(), mlir::hivm::RoundModeAttr::get(&context, mode),
        nullptr);
    return newCastOp->getResult(0);
  };
  
  mlir::Value cond_data_name = GetVarValue(npuirop.cond);
  mlir::Value src0_data_name = GetVarValue(npuirop.src0);
  mlir::Value src1_data_name = GetVarValue(npuirop.src1);
  mlir::Value dst_data_name = GetVarValue(npuirop.dst);

  // 只处理 tensor 场景
  if (dst_data_name.getType().isa<mlir::TensorType>()) {

    auto [dst_offsets, dst_sizes, dst_strides] =
    createOpFoldResultArray(npuirop.dst_range);

    auto srcTensorTy = src0_data_name.getType().cast<mlir::TensorType>();
    auto elemTy = srcTensorTy.getElementType();

    // elementwise shape 来自 src0
    llvm::SmallVector<int64_t> elemShape(srcTensorTy.getShape().begin(),
                                        srcTensorTy.getShape().end());
    
    std::vector<int64_t> elemShapeVec(srcTensorTy.getShape().begin(),
                                        srcTensorTy.getShape().end());

    // auto zeroAttr = builder.getZeroAttr(elemTy);
    auto zeroTensor = builder.create<mlir::tensor::EmptyOp>(
        builder.getUnknownLoc(),
        elemShape,
        elemTy
    );

    auto broadcastDim = getBroadcastDim(npuirop.src0->shape, elemShapeVec);

    auto selOp = builder.create<mlir::hivm::VSelOp>(
        builder.getUnknownLoc(),
        mlir::TypeRange{zeroTensor.getType()},
        mlir::ValueRange{cond_data_name, src0_data_name, src1_data_name},
        mlir::ValueRange{zeroTensor.getResult()},
        mlir::Value() // buffer-style
    );

    selOp->setAttr("broadcast", builder.getDenseI64ArrayAttr(broadcastDim));

    auto reshapeByDstSizes =
      [&](mlir::Value srcTensor,
          llvm::ArrayRef<mlir::OpFoldResult> dst_sizes) -> mlir::Value {

    auto srcType =
        srcTensor.getType().cast<mlir::RankedTensorType>();
    mlir::Type elemTy = srcType.getElementType();
    mlir::Location loc = builder.getUnknownLoc();

    SmallVector<int64_t> targetShape;
    for (auto s : dst_sizes) {
      if (auto attr = s.dyn_cast<mlir::Attribute>()) {
        targetShape.push_back(
            attr.cast<mlir::IntegerAttr>().getInt());
      } else {
        ICHECK(false) << "tensor.reshape requires static dst_sizes";
      }
    }

    if (srcType.hasStaticShape()) {
      int64_t srcElems = srcType.getNumElements();
      int64_t dstElems = 1;
      for (auto d : targetShape)
        dstElems *= d;

      ICHECK(srcElems == dstElems)
          << "Illegal reshape: element count mismatch: "
          << srcElems << " vs " << dstElems;
    }

    SmallVector<mlir::Value> shapeVals;
    for (int64_t d : targetShape) {
      shapeVals.push_back(
          builder.create<mlir::arith::ConstantIndexOp>(loc, d));
    }

    auto shapeTensorType =
        mlir::RankedTensorType::get(
            {(int64_t)shapeVals.size()},
            builder.getIndexType());

    mlir::Value shapeTensor =
        builder.create<mlir::tensor::FromElementsOp>(
            loc, shapeTensorType, shapeVals);

    auto reshapedType =
        mlir::RankedTensorType::get(targetShape, elemTy);

    return builder
        .create<mlir::tensor::ReshapeOp>(
            loc, reshapedType, srcTensor, shapeTensor)
        .getResult();
  };

    mlir::Value selOutput = selOp.getResult()[0];
    mlir::Value reshaped_src = reshapeByDstSizes(selOutput, dst_sizes);
    mlir::Value casted_src = createCastIfTypeMismatch(reshaped_src, dst_data_name);

    auto result = builder.create<mlir::tensor::InsertSliceOp>(
        builder.getUnknownLoc(),
        reshaped_src,
        dst_data_name,
        dst_offsets,
        dst_sizes,
        dst_strides
    );

  SetVarValue(npuirop.dst, result.getResult());
    return;
  }
}

