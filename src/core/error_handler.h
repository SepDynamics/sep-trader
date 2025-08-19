#ifndef SEP_CORE_ERROR_HANDLER_H
#define SEP_CORE_ERROR_HANDLER_H

#include "core/common.h"
#include "core/standard_includes.h"
#include "core/result_types.h"
// Removed duplicate include of core/types.h

namespace sep {

// Error type is now defined in result_types.h to avoid duplication

namespace core {

class ErrorHandler {
 public:
  static ErrorHandler &instance();

  void reportError(const sep::Error &error, std::function<bool()> retry = {});

  std::vector<::sep::Error> getErrors() const;

  void clearErrors();

  bool hasErrors() const;

 private:
  struct Entry {
    ::sep::Error error;
    std::function<bool()> retry;
    std::uint32_t attempts{0};
  };

  ErrorHandler() = default;
  void processRetriesLocked();

  mutable std::mutex mutex_;
  std::vector<Entry> errors_;
};

}  // namespace core
}  // namespace sep

#endif  // SEP_CORE_ERROR_HANDLER_H
