#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Rewrite/Frontend/FixItRewriter.h"

using namespace clang::ast_matchers;
using namespace llvm;
using namespace clang;


class FixItRewriterOptions : public clang::FixItOptions {
 public:
  explicit FixItRewriterOptions(const std::string& RewriteSuffix)
  : RewriteSuffix(RewriteSuffix) {
    // super::InPlace = false;
  }

  /// For a file to be rewritten, returns the (possibly) new filename.
  ///
  /// If the \c RewriteSuffix is empty, returns the \p Filename, causing
  /// in-place rewriting. If it is not empty, the \p Filename with that suffix
  /// is returned.
  std::string RewriteFilename(const std::string& Filename, int& fd) override {
    fd = -1;

    llvm::errs() << "Rewriting FixIts ";

    if (RewriteSuffix.empty()) {
      llvm::errs() << "in-place\n";
      return Filename;
    }

    const auto NewFilename = Filename + RewriteSuffix;
    llvm::errs() << "from " << Filename << " to " << NewFilename << "\n";

    return NewFilename;
  }

 private:
  /// The suffix appended to rewritten files.
  std::string RewriteSuffix;
};


class ClassMemberExprHandler: public MatchFinder::MatchCallback
{
public:
	using RewriterPointer = std::unique_ptr<clang::FixItRewriter>;
	ClassMemberExprHandler(): FixItOptions(""), DoRewrite(true) {};

	virtual void run(const MatchFinder::MatchResult &Result)
	{
		if (const auto* ClassMemberExpr = Result.Nodes.getNodeAs<clang::MemberExpr>("classMemberExpr"))
		{
			auto& Context = *Result.Context;
			auto name = ClassMemberExpr->getMemberNameInfo().getAsString();
			if (name.substr(0,2) == "m_") return;
			DiagnosticsEngine& engine = Result.Context->getDiagnostics();
			
		    RewriterPointer Rewriter;
		    if (DoRewrite) {
		      Rewriter = createRewriter(engine, Context);
		    }

			const unsigned ID = engine.getCustomDiagID(DiagnosticsEngine::Warning,
				"class member variable should have a m_ prefix");
			const FixItHint FixIt = 
			FixItHint::CreateInsertion(ClassMemberExpr->getMemberLoc(), "m_");
			engine.Report(ClassMemberExpr->getMemberLoc(), ID).AddFixItHint(FixIt);
		    if (DoRewrite) {
		      assert(Rewriter != nullptr);
		      Rewriter->WriteFixedFiles();
		    }
		}
	}
private:
  		RewriterPointer createRewriter(clang::DiagnosticsEngine& DiagnosticsEngine,
                                 clang::ASTContext& Context) {
	   		auto Rewriter =
	        std::make_unique<clang::FixItRewriter>(DiagnosticsEngine,
	                                               Context.getSourceManager(),
	                                               Context.getLangOpts(),
	                                               &FixItOptions);

	    	DiagnosticsEngine.setClient(Rewriter.get(), /*ShouldOwnClient=*/false);

	    	return Rewriter;
    	}
	  FixItRewriterOptions FixItOptions;
	  bool DoRewrite;
};


class ClassMemberDeclarationHandler: public MatchFinder::MatchCallback
{
public:
	using RewriterPointer = std::unique_ptr<clang::FixItRewriter>;
	ClassMemberDeclarationHandler(): FixItOptions(""), DoRewrite(true) {};
	virtual void run(const MatchFinder::MatchResult &Result)
	{
		if (const auto* ClassMemberDeclaration = Result.Nodes.getNodeAs<clang::NamedDecl>("classMemberDecl"))
		{
			auto& Context = *Result.Context;
			auto name = ClassMemberDeclaration->getName();
			if (name.startswith("m_")) return;
			DiagnosticsEngine& engine = Result.Context->getDiagnostics();

		    RewriterPointer Rewriter;
		    if (DoRewrite) {
		      Rewriter = createRewriter(engine, Context);
		    }

			const unsigned ID = engine.getCustomDiagID(DiagnosticsEngine::Warning,
				"class member variables should have a m_ prefix");
			const clang::FixItHint FixIt = 
			FixItHint::CreateInsertion(ClassMemberDeclaration->getLocation(), "m_");
			engine.Report(ClassMemberDeclaration->getLocation(), ID).AddFixItHint(FixIt);

		    if (DoRewrite) {
		      assert(Rewriter != nullptr);
		      Rewriter->WriteFixedFiles();
		    }
		}
	}

	 private:
  		RewriterPointer createRewriter(clang::DiagnosticsEngine& DiagnosticsEngine,
                                 clang::ASTContext& Context) {
	   		auto Rewriter =
	        std::make_unique<clang::FixItRewriter>(DiagnosticsEngine,
	                                               Context.getSourceManager(),
	                                               Context.getLangOpts(),
	                                               &FixItOptions);

	    	DiagnosticsEngine.setClient(Rewriter.get(), /*ShouldOwnClient=*/false);

	    	return Rewriter;
    	}

	  FixItRewriterOptions FixItOptions;
	  bool DoRewrite;
};

class Consumer: public clang::ASTConsumer {
public:
	void HandleTranslationUnit(clang::ASTContext& ctx)
	{
		ClassMemberDeclarationHandler Handler1;
		ClassMemberExprHandler Handler2;
		MatchFinder Finder;
		DeclarationMatcher ClassMemberMatcher = fieldDecl().bind("classMemberDecl");
		StatementMatcher ClassMemberExprMatcher = memberExpr().bind("classMemberExpr");
		Finder.addMatcher(ClassMemberMatcher, &Handler1);
		Finder.addMatcher(ClassMemberExprMatcher, &Handler2);
		Finder.matchAST(ctx);
	}
};

namespace classMember_test {
class Action: public clang::ASTFrontendAction {
	public:
	std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(CompilerInstance&, StringRef) override
	{
		return std::make_unique<Consumer>();
	}	
};
}

struct ToolFactory: public clang::tooling::FrontendActionFactory
{
	clang::FrontendAction* create() override
	{
		return new classMember_test::Action();
	}
};

static cl::OptionCategory MyToolCategory("adds m_ prefixs to class members"); 
static cl::extrahelp CommonHelp(tooling::CommonOptionsParser::HelpMessage); 
static cl::extrahelp MoreHelp("\nyup, that's all it does!\n"); 

int main(int argc, const char **argv) {
  tooling::CommonOptionsParser OptionsParser(argc, argv, MyToolCategory);
  tooling::ClangTool Tool(OptionsParser.getCompilations(),
                 OptionsParser.getSourcePathList());
  return Tool.run(new ToolFactory);
}