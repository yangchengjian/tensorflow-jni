// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/example/example.proto

#define INTERNAL_SUPPRESS_PROTOBUF_FIELD_DEPRECATION
#include "tensorflow/core/example/example.pb.h"

#include <algorithm>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/stubs/once.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/wire_format_lite_inl.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)

namespace tensorflow {

namespace {

const ::google::protobuf::Descriptor* Example_descriptor_ = NULL;
const ::google::protobuf::internal::GeneratedMessageReflection*
  Example_reflection_ = NULL;
const ::google::protobuf::Descriptor* SequenceExample_descriptor_ = NULL;
const ::google::protobuf::internal::GeneratedMessageReflection*
  SequenceExample_reflection_ = NULL;
const ::google::protobuf::Descriptor* InferenceExample_descriptor_ = NULL;
const ::google::protobuf::internal::GeneratedMessageReflection*
  InferenceExample_reflection_ = NULL;

}  // namespace


void protobuf_AssignDesc_tensorflow_2fcore_2fexample_2fexample_2eproto() {
  protobuf_AddDesc_tensorflow_2fcore_2fexample_2fexample_2eproto();
  const ::google::protobuf::FileDescriptor* file =
    ::google::protobuf::DescriptorPool::generated_pool()->FindFileByName(
      "tensorflow/core/example/example.proto");
  GOOGLE_CHECK(file != NULL);
  Example_descriptor_ = file->message_type(0);
  static const int Example_offsets_[1] = {
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Example, features_),
  };
  Example_reflection_ =
    ::google::protobuf::internal::GeneratedMessageReflection::NewGeneratedMessageReflection(
      Example_descriptor_,
      Example::default_instance_,
      Example_offsets_,
      -1,
      -1,
      -1,
      sizeof(Example),
      GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Example, _internal_metadata_),
      GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Example, _is_default_instance_));
  SequenceExample_descriptor_ = file->message_type(1);
  static const int SequenceExample_offsets_[2] = {
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(SequenceExample, context_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(SequenceExample, feature_lists_),
  };
  SequenceExample_reflection_ =
    ::google::protobuf::internal::GeneratedMessageReflection::NewGeneratedMessageReflection(
      SequenceExample_descriptor_,
      SequenceExample::default_instance_,
      SequenceExample_offsets_,
      -1,
      -1,
      -1,
      sizeof(SequenceExample),
      GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(SequenceExample, _internal_metadata_),
      GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(SequenceExample, _is_default_instance_));
  InferenceExample_descriptor_ = file->message_type(2);
  static const int InferenceExample_offsets_[2] = {
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(InferenceExample, context_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(InferenceExample, features_),
  };
  InferenceExample_reflection_ =
    ::google::protobuf::internal::GeneratedMessageReflection::NewGeneratedMessageReflection(
      InferenceExample_descriptor_,
      InferenceExample::default_instance_,
      InferenceExample_offsets_,
      -1,
      -1,
      -1,
      sizeof(InferenceExample),
      GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(InferenceExample, _internal_metadata_),
      GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(InferenceExample, _is_default_instance_));
}

namespace {

GOOGLE_PROTOBUF_DECLARE_ONCE(protobuf_AssignDescriptors_once_);
inline void protobuf_AssignDescriptorsOnce() {
  ::google::protobuf::GoogleOnceInit(&protobuf_AssignDescriptors_once_,
                 &protobuf_AssignDesc_tensorflow_2fcore_2fexample_2fexample_2eproto);
}

void protobuf_RegisterTypes(const ::std::string&) {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedMessage(
      Example_descriptor_, &Example::default_instance());
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedMessage(
      SequenceExample_descriptor_, &SequenceExample::default_instance());
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedMessage(
      InferenceExample_descriptor_, &InferenceExample::default_instance());
}

}  // namespace

void protobuf_ShutdownFile_tensorflow_2fcore_2fexample_2fexample_2eproto() {
  delete Example::default_instance_;
  delete Example_reflection_;
  delete SequenceExample::default_instance_;
  delete SequenceExample_reflection_;
  delete InferenceExample::default_instance_;
  delete InferenceExample_reflection_;
}

void protobuf_AddDesc_tensorflow_2fcore_2fexample_2fexample_2eproto() {
  static bool already_here = false;
  if (already_here) return;
  already_here = true;
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  ::tensorflow::protobuf_AddDesc_tensorflow_2fcore_2fexample_2ffeature_2eproto();
  ::google::protobuf::DescriptorPool::InternalAddGeneratedFile(
    "\n%tensorflow/core/example/example.proto\022"
    "\ntensorflow\032%tensorflow/core/example/fea"
    "ture.proto\"1\n\007Example\022&\n\010features\030\001 \001(\0132"
    "\024.tensorflow.Features\"i\n\017SequenceExample"
    "\022%\n\007context\030\001 \001(\0132\024.tensorflow.Features\022"
    "/\n\rfeature_lists\030\002 \001(\0132\030.tensorflow.Feat"
    "ureLists\"a\n\020InferenceExample\022%\n\007context\030"
    "\001 \001(\0132\024.tensorflow.Features\022&\n\010features\030"
    "\002 \003(\0132\024.tensorflow.Featuresb\006proto3", 355);
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedFile(
    "tensorflow/core/example/example.proto", &protobuf_RegisterTypes);
  Example::default_instance_ = new Example();
  SequenceExample::default_instance_ = new SequenceExample();
  InferenceExample::default_instance_ = new InferenceExample();
  Example::default_instance_->InitAsDefaultInstance();
  SequenceExample::default_instance_->InitAsDefaultInstance();
  InferenceExample::default_instance_->InitAsDefaultInstance();
  ::google::protobuf::internal::OnShutdown(&protobuf_ShutdownFile_tensorflow_2fcore_2fexample_2fexample_2eproto);
}

// Force AddDescriptors() to be called at static initialization time.
struct StaticDescriptorInitializer_tensorflow_2fcore_2fexample_2fexample_2eproto {
  StaticDescriptorInitializer_tensorflow_2fcore_2fexample_2fexample_2eproto() {
    protobuf_AddDesc_tensorflow_2fcore_2fexample_2fexample_2eproto();
  }
} static_descriptor_initializer_tensorflow_2fcore_2fexample_2fexample_2eproto_;

namespace {

static void MergeFromFail(int line) GOOGLE_ATTRIBUTE_COLD;
static void MergeFromFail(int line) {
  GOOGLE_CHECK(false) << __FILE__ << ":" << line;
}

}  // namespace


// ===================================================================

#ifndef _MSC_VER
const int Example::kFeaturesFieldNumber;
#endif  // !_MSC_VER

Example::Example()
  : ::google::protobuf::Message(), _internal_metadata_(NULL) {
  SharedCtor();
  // @@protoc_insertion_point(constructor:tensorflow.Example)
}

void Example::InitAsDefaultInstance() {
  _is_default_instance_ = true;
  features_ = const_cast< ::tensorflow::Features*>(&::tensorflow::Features::default_instance());
}

Example::Example(const Example& from)
  : ::google::protobuf::Message(),
    _internal_metadata_(NULL) {
  SharedCtor();
  MergeFrom(from);
  // @@protoc_insertion_point(copy_constructor:tensorflow.Example)
}

void Example::SharedCtor() {
    _is_default_instance_ = false;
  _cached_size_ = 0;
  features_ = NULL;
}

Example::~Example() {
  // @@protoc_insertion_point(destructor:tensorflow.Example)
  SharedDtor();
}

void Example::SharedDtor() {
  if (this != default_instance_) {
    delete features_;
  }
}

void Example::SetCachedSize(int size) const {
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
}
const ::google::protobuf::Descriptor* Example::descriptor() {
  protobuf_AssignDescriptorsOnce();
  return Example_descriptor_;
}

const Example& Example::default_instance() {
  if (default_instance_ == NULL) protobuf_AddDesc_tensorflow_2fcore_2fexample_2fexample_2eproto();
  return *default_instance_;
}

Example* Example::default_instance_ = NULL;

Example* Example::New(::google::protobuf::Arena* arena) const {
  Example* n = new Example;
  if (arena != NULL) {
    arena->Own(n);
  }
  return n;
}

void Example::Clear() {
  if (GetArenaNoVirtual() == NULL && features_ != NULL) delete features_;
  features_ = NULL;
}

bool Example::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!(EXPRESSION)) goto failure
  ::google::protobuf::uint32 tag;
  // @@protoc_insertion_point(parse_start:tensorflow.Example)
  for (;;) {
    ::std::pair< ::google::protobuf::uint32, bool> p = input->ReadTagWithCutoff(127);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // optional .tensorflow.Features features = 1;
      case 1: {
        if (tag == 10) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadMessageNoVirtual(
               input, mutable_features()));
        } else {
          goto handle_unusual;
        }
        if (input->ExpectAtEnd()) goto success;
        break;
      }

      default: {
      handle_unusual:
        if (tag == 0 ||
            ::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_END_GROUP) {
          goto success;
        }
        DO_(::google::protobuf::internal::WireFormatLite::SkipField(input, tag));
        break;
      }
    }
  }
success:
  // @@protoc_insertion_point(parse_success:tensorflow.Example)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:tensorflow.Example)
  return false;
#undef DO_
}

void Example::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:tensorflow.Example)
  // optional .tensorflow.Features features = 1;
  if (this->has_features()) {
    ::google::protobuf::internal::WireFormatLite::WriteMessageMaybeToArray(
      1, *this->features_, output);
  }

  // @@protoc_insertion_point(serialize_end:tensorflow.Example)
}

::google::protobuf::uint8* Example::SerializeWithCachedSizesToArray(
    ::google::protobuf::uint8* target) const {
  // @@protoc_insertion_point(serialize_to_array_start:tensorflow.Example)
  // optional .tensorflow.Features features = 1;
  if (this->has_features()) {
    target = ::google::protobuf::internal::WireFormatLite::
      WriteMessageNoVirtualToArray(
        1, *this->features_, target);
  }

  // @@protoc_insertion_point(serialize_to_array_end:tensorflow.Example)
  return target;
}

int Example::ByteSize() const {
  int total_size = 0;

  // optional .tensorflow.Features features = 1;
  if (this->has_features()) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::MessageSizeNoVirtual(
        *this->features_);
  }

  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = total_size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
  return total_size;
}

void Example::MergeFrom(const ::google::protobuf::Message& from) {
  if (GOOGLE_PREDICT_FALSE(&from == this)) MergeFromFail(__LINE__);
  const Example* source = 
      ::google::protobuf::internal::DynamicCastToGenerated<const Example>(
          &from);
  if (source == NULL) {
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
    MergeFrom(*source);
  }
}

void Example::MergeFrom(const Example& from) {
  if (GOOGLE_PREDICT_FALSE(&from == this)) MergeFromFail(__LINE__);
  if (from.has_features()) {
    mutable_features()->::tensorflow::Features::MergeFrom(from.features());
  }
}

void Example::CopyFrom(const ::google::protobuf::Message& from) {
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void Example::CopyFrom(const Example& from) {
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool Example::IsInitialized() const {

  return true;
}

void Example::Swap(Example* other) {
  if (other == this) return;
  InternalSwap(other);
}
void Example::InternalSwap(Example* other) {
  std::swap(features_, other->features_);
  _internal_metadata_.Swap(&other->_internal_metadata_);
  std::swap(_cached_size_, other->_cached_size_);
}

::google::protobuf::Metadata Example::GetMetadata() const {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::Metadata metadata;
  metadata.descriptor = Example_descriptor_;
  metadata.reflection = Example_reflection_;
  return metadata;
}

#if PROTOBUF_INLINE_NOT_IN_HEADERS
// Example

// optional .tensorflow.Features features = 1;
bool Example::has_features() const {
  return !_is_default_instance_ && features_ != NULL;
}
void Example::clear_features() {
  if (GetArenaNoVirtual() == NULL && features_ != NULL) delete features_;
  features_ = NULL;
}
const ::tensorflow::Features& Example::features() const {
  // @@protoc_insertion_point(field_get:tensorflow.Example.features)
  return features_ != NULL ? *features_ : *default_instance_->features_;
}
::tensorflow::Features* Example::mutable_features() {
  
  if (features_ == NULL) {
    features_ = new ::tensorflow::Features;
  }
  // @@protoc_insertion_point(field_mutable:tensorflow.Example.features)
  return features_;
}
::tensorflow::Features* Example::release_features() {
  
  ::tensorflow::Features* temp = features_;
  features_ = NULL;
  return temp;
}
void Example::set_allocated_features(::tensorflow::Features* features) {
  delete features_;
  features_ = features;
  if (features) {
    
  } else {
    
  }
  // @@protoc_insertion_point(field_set_allocated:tensorflow.Example.features)
}

#endif  // PROTOBUF_INLINE_NOT_IN_HEADERS

// ===================================================================

#ifndef _MSC_VER
const int SequenceExample::kContextFieldNumber;
const int SequenceExample::kFeatureListsFieldNumber;
#endif  // !_MSC_VER

SequenceExample::SequenceExample()
  : ::google::protobuf::Message(), _internal_metadata_(NULL) {
  SharedCtor();
  // @@protoc_insertion_point(constructor:tensorflow.SequenceExample)
}

void SequenceExample::InitAsDefaultInstance() {
  _is_default_instance_ = true;
  context_ = const_cast< ::tensorflow::Features*>(&::tensorflow::Features::default_instance());
  feature_lists_ = const_cast< ::tensorflow::FeatureLists*>(&::tensorflow::FeatureLists::default_instance());
}

SequenceExample::SequenceExample(const SequenceExample& from)
  : ::google::protobuf::Message(),
    _internal_metadata_(NULL) {
  SharedCtor();
  MergeFrom(from);
  // @@protoc_insertion_point(copy_constructor:tensorflow.SequenceExample)
}

void SequenceExample::SharedCtor() {
    _is_default_instance_ = false;
  _cached_size_ = 0;
  context_ = NULL;
  feature_lists_ = NULL;
}

SequenceExample::~SequenceExample() {
  // @@protoc_insertion_point(destructor:tensorflow.SequenceExample)
  SharedDtor();
}

void SequenceExample::SharedDtor() {
  if (this != default_instance_) {
    delete context_;
    delete feature_lists_;
  }
}

void SequenceExample::SetCachedSize(int size) const {
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
}
const ::google::protobuf::Descriptor* SequenceExample::descriptor() {
  protobuf_AssignDescriptorsOnce();
  return SequenceExample_descriptor_;
}

const SequenceExample& SequenceExample::default_instance() {
  if (default_instance_ == NULL) protobuf_AddDesc_tensorflow_2fcore_2fexample_2fexample_2eproto();
  return *default_instance_;
}

SequenceExample* SequenceExample::default_instance_ = NULL;

SequenceExample* SequenceExample::New(::google::protobuf::Arena* arena) const {
  SequenceExample* n = new SequenceExample;
  if (arena != NULL) {
    arena->Own(n);
  }
  return n;
}

void SequenceExample::Clear() {
  if (GetArenaNoVirtual() == NULL && context_ != NULL) delete context_;
  context_ = NULL;
  if (GetArenaNoVirtual() == NULL && feature_lists_ != NULL) delete feature_lists_;
  feature_lists_ = NULL;
}

bool SequenceExample::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!(EXPRESSION)) goto failure
  ::google::protobuf::uint32 tag;
  // @@protoc_insertion_point(parse_start:tensorflow.SequenceExample)
  for (;;) {
    ::std::pair< ::google::protobuf::uint32, bool> p = input->ReadTagWithCutoff(127);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // optional .tensorflow.Features context = 1;
      case 1: {
        if (tag == 10) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadMessageNoVirtual(
               input, mutable_context()));
        } else {
          goto handle_unusual;
        }
        if (input->ExpectTag(18)) goto parse_feature_lists;
        break;
      }

      // optional .tensorflow.FeatureLists feature_lists = 2;
      case 2: {
        if (tag == 18) {
         parse_feature_lists:
          DO_(::google::protobuf::internal::WireFormatLite::ReadMessageNoVirtual(
               input, mutable_feature_lists()));
        } else {
          goto handle_unusual;
        }
        if (input->ExpectAtEnd()) goto success;
        break;
      }

      default: {
      handle_unusual:
        if (tag == 0 ||
            ::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_END_GROUP) {
          goto success;
        }
        DO_(::google::protobuf::internal::WireFormatLite::SkipField(input, tag));
        break;
      }
    }
  }
success:
  // @@protoc_insertion_point(parse_success:tensorflow.SequenceExample)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:tensorflow.SequenceExample)
  return false;
#undef DO_
}

void SequenceExample::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:tensorflow.SequenceExample)
  // optional .tensorflow.Features context = 1;
  if (this->has_context()) {
    ::google::protobuf::internal::WireFormatLite::WriteMessageMaybeToArray(
      1, *this->context_, output);
  }

  // optional .tensorflow.FeatureLists feature_lists = 2;
  if (this->has_feature_lists()) {
    ::google::protobuf::internal::WireFormatLite::WriteMessageMaybeToArray(
      2, *this->feature_lists_, output);
  }

  // @@protoc_insertion_point(serialize_end:tensorflow.SequenceExample)
}

::google::protobuf::uint8* SequenceExample::SerializeWithCachedSizesToArray(
    ::google::protobuf::uint8* target) const {
  // @@protoc_insertion_point(serialize_to_array_start:tensorflow.SequenceExample)
  // optional .tensorflow.Features context = 1;
  if (this->has_context()) {
    target = ::google::protobuf::internal::WireFormatLite::
      WriteMessageNoVirtualToArray(
        1, *this->context_, target);
  }

  // optional .tensorflow.FeatureLists feature_lists = 2;
  if (this->has_feature_lists()) {
    target = ::google::protobuf::internal::WireFormatLite::
      WriteMessageNoVirtualToArray(
        2, *this->feature_lists_, target);
  }

  // @@protoc_insertion_point(serialize_to_array_end:tensorflow.SequenceExample)
  return target;
}

int SequenceExample::ByteSize() const {
  int total_size = 0;

  // optional .tensorflow.Features context = 1;
  if (this->has_context()) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::MessageSizeNoVirtual(
        *this->context_);
  }

  // optional .tensorflow.FeatureLists feature_lists = 2;
  if (this->has_feature_lists()) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::MessageSizeNoVirtual(
        *this->feature_lists_);
  }

  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = total_size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
  return total_size;
}

void SequenceExample::MergeFrom(const ::google::protobuf::Message& from) {
  if (GOOGLE_PREDICT_FALSE(&from == this)) MergeFromFail(__LINE__);
  const SequenceExample* source = 
      ::google::protobuf::internal::DynamicCastToGenerated<const SequenceExample>(
          &from);
  if (source == NULL) {
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
    MergeFrom(*source);
  }
}

void SequenceExample::MergeFrom(const SequenceExample& from) {
  if (GOOGLE_PREDICT_FALSE(&from == this)) MergeFromFail(__LINE__);
  if (from.has_context()) {
    mutable_context()->::tensorflow::Features::MergeFrom(from.context());
  }
  if (from.has_feature_lists()) {
    mutable_feature_lists()->::tensorflow::FeatureLists::MergeFrom(from.feature_lists());
  }
}

void SequenceExample::CopyFrom(const ::google::protobuf::Message& from) {
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void SequenceExample::CopyFrom(const SequenceExample& from) {
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool SequenceExample::IsInitialized() const {

  return true;
}

void SequenceExample::Swap(SequenceExample* other) {
  if (other == this) return;
  InternalSwap(other);
}
void SequenceExample::InternalSwap(SequenceExample* other) {
  std::swap(context_, other->context_);
  std::swap(feature_lists_, other->feature_lists_);
  _internal_metadata_.Swap(&other->_internal_metadata_);
  std::swap(_cached_size_, other->_cached_size_);
}

::google::protobuf::Metadata SequenceExample::GetMetadata() const {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::Metadata metadata;
  metadata.descriptor = SequenceExample_descriptor_;
  metadata.reflection = SequenceExample_reflection_;
  return metadata;
}

#if PROTOBUF_INLINE_NOT_IN_HEADERS
// SequenceExample

// optional .tensorflow.Features context = 1;
bool SequenceExample::has_context() const {
  return !_is_default_instance_ && context_ != NULL;
}
void SequenceExample::clear_context() {
  if (GetArenaNoVirtual() == NULL && context_ != NULL) delete context_;
  context_ = NULL;
}
const ::tensorflow::Features& SequenceExample::context() const {
  // @@protoc_insertion_point(field_get:tensorflow.SequenceExample.context)
  return context_ != NULL ? *context_ : *default_instance_->context_;
}
::tensorflow::Features* SequenceExample::mutable_context() {
  
  if (context_ == NULL) {
    context_ = new ::tensorflow::Features;
  }
  // @@protoc_insertion_point(field_mutable:tensorflow.SequenceExample.context)
  return context_;
}
::tensorflow::Features* SequenceExample::release_context() {
  
  ::tensorflow::Features* temp = context_;
  context_ = NULL;
  return temp;
}
void SequenceExample::set_allocated_context(::tensorflow::Features* context) {
  delete context_;
  context_ = context;
  if (context) {
    
  } else {
    
  }
  // @@protoc_insertion_point(field_set_allocated:tensorflow.SequenceExample.context)
}

// optional .tensorflow.FeatureLists feature_lists = 2;
bool SequenceExample::has_feature_lists() const {
  return !_is_default_instance_ && feature_lists_ != NULL;
}
void SequenceExample::clear_feature_lists() {
  if (GetArenaNoVirtual() == NULL && feature_lists_ != NULL) delete feature_lists_;
  feature_lists_ = NULL;
}
const ::tensorflow::FeatureLists& SequenceExample::feature_lists() const {
  // @@protoc_insertion_point(field_get:tensorflow.SequenceExample.feature_lists)
  return feature_lists_ != NULL ? *feature_lists_ : *default_instance_->feature_lists_;
}
::tensorflow::FeatureLists* SequenceExample::mutable_feature_lists() {
  
  if (feature_lists_ == NULL) {
    feature_lists_ = new ::tensorflow::FeatureLists;
  }
  // @@protoc_insertion_point(field_mutable:tensorflow.SequenceExample.feature_lists)
  return feature_lists_;
}
::tensorflow::FeatureLists* SequenceExample::release_feature_lists() {
  
  ::tensorflow::FeatureLists* temp = feature_lists_;
  feature_lists_ = NULL;
  return temp;
}
void SequenceExample::set_allocated_feature_lists(::tensorflow::FeatureLists* feature_lists) {
  delete feature_lists_;
  feature_lists_ = feature_lists;
  if (feature_lists) {
    
  } else {
    
  }
  // @@protoc_insertion_point(field_set_allocated:tensorflow.SequenceExample.feature_lists)
}

#endif  // PROTOBUF_INLINE_NOT_IN_HEADERS

// ===================================================================

#ifndef _MSC_VER
const int InferenceExample::kContextFieldNumber;
const int InferenceExample::kFeaturesFieldNumber;
#endif  // !_MSC_VER

InferenceExample::InferenceExample()
  : ::google::protobuf::Message(), _internal_metadata_(NULL) {
  SharedCtor();
  // @@protoc_insertion_point(constructor:tensorflow.InferenceExample)
}

void InferenceExample::InitAsDefaultInstance() {
  _is_default_instance_ = true;
  context_ = const_cast< ::tensorflow::Features*>(&::tensorflow::Features::default_instance());
}

InferenceExample::InferenceExample(const InferenceExample& from)
  : ::google::protobuf::Message(),
    _internal_metadata_(NULL) {
  SharedCtor();
  MergeFrom(from);
  // @@protoc_insertion_point(copy_constructor:tensorflow.InferenceExample)
}

void InferenceExample::SharedCtor() {
    _is_default_instance_ = false;
  _cached_size_ = 0;
  context_ = NULL;
}

InferenceExample::~InferenceExample() {
  // @@protoc_insertion_point(destructor:tensorflow.InferenceExample)
  SharedDtor();
}

void InferenceExample::SharedDtor() {
  if (this != default_instance_) {
    delete context_;
  }
}

void InferenceExample::SetCachedSize(int size) const {
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
}
const ::google::protobuf::Descriptor* InferenceExample::descriptor() {
  protobuf_AssignDescriptorsOnce();
  return InferenceExample_descriptor_;
}

const InferenceExample& InferenceExample::default_instance() {
  if (default_instance_ == NULL) protobuf_AddDesc_tensorflow_2fcore_2fexample_2fexample_2eproto();
  return *default_instance_;
}

InferenceExample* InferenceExample::default_instance_ = NULL;

InferenceExample* InferenceExample::New(::google::protobuf::Arena* arena) const {
  InferenceExample* n = new InferenceExample;
  if (arena != NULL) {
    arena->Own(n);
  }
  return n;
}

void InferenceExample::Clear() {
  if (GetArenaNoVirtual() == NULL && context_ != NULL) delete context_;
  context_ = NULL;
  features_.Clear();
}

bool InferenceExample::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!(EXPRESSION)) goto failure
  ::google::protobuf::uint32 tag;
  // @@protoc_insertion_point(parse_start:tensorflow.InferenceExample)
  for (;;) {
    ::std::pair< ::google::protobuf::uint32, bool> p = input->ReadTagWithCutoff(127);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // optional .tensorflow.Features context = 1;
      case 1: {
        if (tag == 10) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadMessageNoVirtual(
               input, mutable_context()));
        } else {
          goto handle_unusual;
        }
        if (input->ExpectTag(18)) goto parse_features;
        break;
      }

      // repeated .tensorflow.Features features = 2;
      case 2: {
        if (tag == 18) {
         parse_features:
          DO_(input->IncrementRecursionDepth());
         parse_loop_features:
          DO_(::google::protobuf::internal::WireFormatLite::ReadMessageNoVirtualNoRecursionDepth(
                input, add_features()));
        } else {
          goto handle_unusual;
        }
        if (input->ExpectTag(18)) goto parse_loop_features;
        input->UnsafeDecrementRecursionDepth();
        if (input->ExpectAtEnd()) goto success;
        break;
      }

      default: {
      handle_unusual:
        if (tag == 0 ||
            ::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_END_GROUP) {
          goto success;
        }
        DO_(::google::protobuf::internal::WireFormatLite::SkipField(input, tag));
        break;
      }
    }
  }
success:
  // @@protoc_insertion_point(parse_success:tensorflow.InferenceExample)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:tensorflow.InferenceExample)
  return false;
#undef DO_
}

void InferenceExample::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:tensorflow.InferenceExample)
  // optional .tensorflow.Features context = 1;
  if (this->has_context()) {
    ::google::protobuf::internal::WireFormatLite::WriteMessageMaybeToArray(
      1, *this->context_, output);
  }

  // repeated .tensorflow.Features features = 2;
  for (unsigned int i = 0, n = this->features_size(); i < n; i++) {
    ::google::protobuf::internal::WireFormatLite::WriteMessageMaybeToArray(
      2, this->features(i), output);
  }

  // @@protoc_insertion_point(serialize_end:tensorflow.InferenceExample)
}

::google::protobuf::uint8* InferenceExample::SerializeWithCachedSizesToArray(
    ::google::protobuf::uint8* target) const {
  // @@protoc_insertion_point(serialize_to_array_start:tensorflow.InferenceExample)
  // optional .tensorflow.Features context = 1;
  if (this->has_context()) {
    target = ::google::protobuf::internal::WireFormatLite::
      WriteMessageNoVirtualToArray(
        1, *this->context_, target);
  }

  // repeated .tensorflow.Features features = 2;
  for (unsigned int i = 0, n = this->features_size(); i < n; i++) {
    target = ::google::protobuf::internal::WireFormatLite::
      WriteMessageNoVirtualToArray(
        2, this->features(i), target);
  }

  // @@protoc_insertion_point(serialize_to_array_end:tensorflow.InferenceExample)
  return target;
}

int InferenceExample::ByteSize() const {
  int total_size = 0;

  // optional .tensorflow.Features context = 1;
  if (this->has_context()) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::MessageSizeNoVirtual(
        *this->context_);
  }

  // repeated .tensorflow.Features features = 2;
  total_size += 1 * this->features_size();
  for (int i = 0; i < this->features_size(); i++) {
    total_size +=
      ::google::protobuf::internal::WireFormatLite::MessageSizeNoVirtual(
        this->features(i));
  }

  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = total_size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
  return total_size;
}

void InferenceExample::MergeFrom(const ::google::protobuf::Message& from) {
  if (GOOGLE_PREDICT_FALSE(&from == this)) MergeFromFail(__LINE__);
  const InferenceExample* source = 
      ::google::protobuf::internal::DynamicCastToGenerated<const InferenceExample>(
          &from);
  if (source == NULL) {
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
    MergeFrom(*source);
  }
}

void InferenceExample::MergeFrom(const InferenceExample& from) {
  if (GOOGLE_PREDICT_FALSE(&from == this)) MergeFromFail(__LINE__);
  features_.MergeFrom(from.features_);
  if (from.has_context()) {
    mutable_context()->::tensorflow::Features::MergeFrom(from.context());
  }
}

void InferenceExample::CopyFrom(const ::google::protobuf::Message& from) {
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void InferenceExample::CopyFrom(const InferenceExample& from) {
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool InferenceExample::IsInitialized() const {

  return true;
}

void InferenceExample::Swap(InferenceExample* other) {
  if (other == this) return;
  InternalSwap(other);
}
void InferenceExample::InternalSwap(InferenceExample* other) {
  std::swap(context_, other->context_);
  features_.UnsafeArenaSwap(&other->features_);
  _internal_metadata_.Swap(&other->_internal_metadata_);
  std::swap(_cached_size_, other->_cached_size_);
}

::google::protobuf::Metadata InferenceExample::GetMetadata() const {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::Metadata metadata;
  metadata.descriptor = InferenceExample_descriptor_;
  metadata.reflection = InferenceExample_reflection_;
  return metadata;
}

#if PROTOBUF_INLINE_NOT_IN_HEADERS
// InferenceExample

// optional .tensorflow.Features context = 1;
bool InferenceExample::has_context() const {
  return !_is_default_instance_ && context_ != NULL;
}
void InferenceExample::clear_context() {
  if (GetArenaNoVirtual() == NULL && context_ != NULL) delete context_;
  context_ = NULL;
}
const ::tensorflow::Features& InferenceExample::context() const {
  // @@protoc_insertion_point(field_get:tensorflow.InferenceExample.context)
  return context_ != NULL ? *context_ : *default_instance_->context_;
}
::tensorflow::Features* InferenceExample::mutable_context() {
  
  if (context_ == NULL) {
    context_ = new ::tensorflow::Features;
  }
  // @@protoc_insertion_point(field_mutable:tensorflow.InferenceExample.context)
  return context_;
}
::tensorflow::Features* InferenceExample::release_context() {
  
  ::tensorflow::Features* temp = context_;
  context_ = NULL;
  return temp;
}
void InferenceExample::set_allocated_context(::tensorflow::Features* context) {
  delete context_;
  context_ = context;
  if (context) {
    
  } else {
    
  }
  // @@protoc_insertion_point(field_set_allocated:tensorflow.InferenceExample.context)
}

// repeated .tensorflow.Features features = 2;
int InferenceExample::features_size() const {
  return features_.size();
}
void InferenceExample::clear_features() {
  features_.Clear();
}
const ::tensorflow::Features& InferenceExample::features(int index) const {
  // @@protoc_insertion_point(field_get:tensorflow.InferenceExample.features)
  return features_.Get(index);
}
::tensorflow::Features* InferenceExample::mutable_features(int index) {
  // @@protoc_insertion_point(field_mutable:tensorflow.InferenceExample.features)
  return features_.Mutable(index);
}
::tensorflow::Features* InferenceExample::add_features() {
  // @@protoc_insertion_point(field_add:tensorflow.InferenceExample.features)
  return features_.Add();
}
::google::protobuf::RepeatedPtrField< ::tensorflow::Features >*
InferenceExample::mutable_features() {
  // @@protoc_insertion_point(field_mutable_list:tensorflow.InferenceExample.features)
  return &features_;
}
const ::google::protobuf::RepeatedPtrField< ::tensorflow::Features >&
InferenceExample::features() const {
  // @@protoc_insertion_point(field_list:tensorflow.InferenceExample.features)
  return features_;
}

#endif  // PROTOBUF_INLINE_NOT_IN_HEADERS

// @@protoc_insertion_point(namespace_scope)

}  // namespace tensorflow

// @@protoc_insertion_point(global_scope)
